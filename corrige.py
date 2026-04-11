from pathlib import Path
import requests
import sys
import re
import json

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "local-model"  # remplace si besoin

SYSTEM_PROMPT = (
    "Tu es un relecteur humain de sous-titres français. "
    "Pour chaque ligne, tu dois produire deux choses : "
    "1) une correction sûre minimale du texte principal ; "
    "2) éventuellement une proposition plus naturelle, plus idiomatique ou plus française. "
    "Règles absolues pour la correction sûre : "
    "ne jamais changer le sens, "
    "ne jamais changer le temps verbal sans nécessité, "
    "ne jamais changer le tutoiement/vouvoiement, "
    "ne jamais changer la personne grammaticale, "
    "ne jamais modifier les tags ASS, "
    "ne jamais modifier les \\N, "
    "ne jamais supprimer les tirets de dialogue. "
    "La correction sûre doit rester prudente. "
    "La proposition QC peut être plus fluide, mais doit garder exactement le même sens. "
    "Réponds uniquement en JSON valide avec ce format exact : "
    '{"corrected":"...","suggestion":"...","reason":"..."}'
)

def parse_dialogue_line(line: str):
    if not line.startswith("Dialogue:"):
        return None
    parts = line.rstrip("\r\n").split(",", 9)
    if len(parts) != 10:
        return None
    return parts

def extract_tags(text: str):
    return re.findall(r"\{.*?\}", text)

def strip_tags(text: str):
    return re.sub(r"\{.*?\}", "", text)

def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("\\N", " ")
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"[^\w\sàâäçéèêëîïôöùûüÿœæ'-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def words(text: str):
    return normalize(text).split()

def similarity_ratio(a: str, b: str) -> float:
    wa = set(words(a))
    wb = set(words(b))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)

def dash_shape(text: str):
    return [chunk.lstrip().startswith("-") for chunk in text.split("\\N")]

def risky_line(text: str) -> bool:
    if r"\p1" in text or r"\p2" in text or r"\p3" in text:
        return True
    if len(strip_tags(text).strip()) <= 1:
        return True
    return False

def local_safe_fixes(text: str) -> str:
    """
    Corrections ultra sûres uniquement.
    Zéro transformation risquée.
    """

    fixes = [
        (r"\bje c\b", "je sais"),
        (r"\bjé\b", "j'ai"),
        (r"\bske\b", "ce que"),
        (r"\bc koi\b", "c'est quoi"),
        (r"\bsa va\b", "ça va"),
        (r"\bquil\b", "qu'il"),
    ]

    for pat, repl in fixes:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # participe passé (cas assez safe)
    text = re.sub(
        r"\bj['’]ai ([a-zàâäçéèêëîïôöùûüÿœæ-]+)er\b",
        r"j'ai \1é",
        text,
        flags=re.IGNORECASE,
    )

    return text

def register_shift(original: str, candidate: str) -> bool:
    o = f" {normalize(original)} "
    c = f" {normalize(candidate)} "

    pairs = [
        (" tu ", " vous "),
        (" vous ", " tu "),
        (" te ", " vous "),
        (" ton ", " votre "),
        (" ta ", " votre "),
        (" tes ", " vos "),
        (" votre ", " ton "),
        (" vos ", " tes "),
        (" ne vous ", " ne te "),
        (" ne te ", " ne vous "),
        (" pardonnez ", " pardonne "),
        (" pardonne ", " pardonnez "),
    ]
    for a, b in pairs:
        if a in o and b in c and a not in c:
            return True
    return False

def tense_shift(original: str, candidate: str) -> bool:
    o = normalize(original)
    c = normalize(candidate)

    suspicious_pairs = [
        ("je me demande", "je m'étais demandé"),
        ("je me disais", "je m'étais dit"),
        ("je commençais", "j'ai commencé"),
        ("je pense", "j'ai pensé"),
        ("je finirai", "j'aurai fini"),
        ("je suis", "j'ai été"),
        ("je préfère", "j'ai préféré"),
        ("on devait", "on n'était pas censés"),
        ("tu veux rentrer", "tu veux entrer"),
    ]
    for a, b in suspicious_pairs:
        if a in o and b in c:
            return True
    return False

def broken_output(text: str) -> bool:
    low = text.lower()
    bad = [
        "{\\n}",
        "{\\N}",
        "{\\c}",
        "{\\anm}",
        "{\\/i1}",
        "j'ai suis",
        "quoi que ce soit",
        "voilà, tard",
    ]
    return any(x.lower() in low for x in bad)

def valid_candidate(original: str, candidate: str, strict: bool = True):
    if not candidate:
        return False, "empty"

    if extract_tags(original) != extract_tags(candidate):
        return False, "tags"

    if original.count("\\N") != candidate.count("\\N"):
        return False, "N"

    if original.endswith("\\N") != candidate.endswith("\\N"):
        return False, "N_end"

    if dash_shape(original) != dash_shape(candidate):
        return False, "dash"

    if broken_output(candidate):
        return False, "broken"

    if register_shift(original, candidate):
        return False, "register"

    if tense_shift(original, candidate):
        return False, "tense"

    threshold = 0.60 if strict else 0.42
    if similarity_ratio(original, candidate) < threshold:
        return False, "different"

    return True, "ok"

def parse_model_json(raw: str):
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return {
            "corrected": str(data.get("corrected", "")).strip(),
            "suggestion": str(data.get("suggestion", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
        }
    except Exception:
        pass

    m = re.search(
        r'\{.*"corrected"\s*:\s*".*?".*"suggestion"\s*:\s*".*?".*"reason"\s*:\s*".*?".*\}',
        raw,
        flags=re.DOTALL,
    )
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "corrected": str(data.get("corrected", "")).strip(),
                "suggestion": str(data.get("suggestion", "")).strip(),
                "reason": str(data.get("reason", "")).strip(),
            }
        except Exception:
            pass

    return None

def call_model(text: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyse cette ligne de sous-titre.\n"
                    "Consignes :\n"
                    "- corrected = correction sûre et prudente\n"
                    "- suggestion = tournure plus naturelle si utile, sinon chaîne vide\n"
                    "- garde les tags ASS, les \\N et les tirets\n"
                    "- conserve exactement le sens\n\n"
                    f"{text}"
                )
            }
        ],
        "temperature": 0.0
    }

    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return parse_model_json(raw)

def make_comment_line(parts, suggestion, reason):
    comment_parts = parts.copy()
    comment_parts[0] = "Comment: 0"
    comment_parts[9] = f"QC: {suggestion}"
    if reason:
        comment_parts[9] += f" [{reason}]"
    return ",".join(comment_parts)

def process_line(text: str):
    original = text

    if risky_line(text):
        return original, None, "ligne spéciale"

    prepared = local_safe_fixes(text)

    result = None
    try:
        result = call_model(prepared)
    except Exception:
        return original, None, "erreur modèle"

    if result is None:
        return original, None, "réponse invalide"

    corrected = result["corrected"] or prepared
    suggestion = result["suggestion"]
    reason = result["reason"]

    ok_corr, why_corr = valid_candidate(original, corrected, strict=True)
    if not ok_corr:
        # on garde éventuellement la pré-correction locale si elle est sûre
        ok_pre, _ = valid_candidate(original, prepared, strict=True)
        if ok_pre and prepared != original:
            corrected = prepared
        else:
            corrected = original

    final_suggestion = None
    if suggestion and suggestion != corrected:
        ok_sugg, _ = valid_candidate(corrected, suggestion, strict=False)
        if ok_sugg:
            final_suggestion = suggestion

    return corrected, final_suggestion, reason

def main():
    if len(sys.argv) < 2:
        print('Usage : py corrige_qc_humain.py "mon_fichier.ass"')
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}")
        return

    original_text = input_path.read_text(encoding="utf-8-sig")
    backup_path = input_path.with_name(input_path.stem + "_backup.ass")
    output_path = input_path.with_name(input_path.stem + "_qc_humain.ass")

    backup_path.write_text(original_text, encoding="utf-8")

    lines = original_text.splitlines()

    changed = 0
    comments = 0
    kept = 0

    for idx, line in enumerate(lines):
        parts = parse_dialogue_line(line)
        if not parts:
            continue

        text = parts[9]
        corrected, suggestion, reason = process_line(text)

        if corrected != text:
            changed += 1
            print(f"[CORRECT] {text} -> {corrected}")
            if reason:
                print(f"  Raison : {reason}")
        else:
            kept += 1
            print(f"[KEEP] {text}")

        parts[9] = corrected
        lines[idx] = ",".join(parts)

        if suggestion:
            comment_line = make_comment_line(parts, suggestion, reason)
            lines.insert(idx + 1, comment_line)
            comments += 1
            print(f"[QC] {corrected} || {suggestion}")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Backup : {backup_path.name}")
    print(f"Sortie : {output_path.name}")
    print(f"Lignes modifiées : {changed}")
    print(f"Lignes avec commentaire QC : {comments}")
    print(f"Lignes conservées : {kept}")

if __name__ == "__main__":
    main()
