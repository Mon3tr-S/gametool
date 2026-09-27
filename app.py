"""
ゲームアイデア生成ツール
AIが「テーマ × 指定された対象」の掛け合わせを10個提示し、
タップするとその場で遊べるデモが生成される。
生成したゲームは saved/ に保存され、APIを呼ばずに再訪できる。
軽い処理は Gemini、コード生成は Claude と使い分けてコストを抑える。
"""

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
SAVED_DIR = BASE_DIR / "saved"
IDEAS_LOG = SAVED_DIR / "ideas-log.jsonl"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ---------- 検証モード ----------
# 環境変数 USE_CLAUDE_CLI=1 にすると、すべてのAI呼び出しを Claude Code CLI
# （定額プランの利用枠）に流し、APIの従量課金を発生させない（要 Claude Code インストール）。
# 未設定なら False ＝ 下の MODEL_* の割り当て（Gemini/Anthropic API）で動く。
USE_CLAUDE_CLI = os.environ.get("USE_CLAUDE_CLI", "").strip().lower() in ("1", "true", "yes", "on")

# ---------- モデル割り当て（用途ごと。あとから差し替え可能） ----------
# 依頼指定の "gemini-3-flash" は実在しないため、現行3系Flashの "gemini-3.7-flash" を使用。

# 軽い処理（短文・JSON出力）
MODEL_ESSENCE = ("gemini", "gemini-3.7-flash")   # 遊びの本質を抽出する
MODEL_IDEAS   = ("gemini", "gemini-3.7-flash")   # アイデア10案
MODEL_TRIAGE  = ("gemini", "gemini-3.7-flash")   # 修正依頼の振り分け
MODEL_CONFIG  = ("gemini", "gemini-3.7-flash")   # CONFIG値の書き換え

# 重い処理（コード生成）
MODEL_GAME    = ("anthropic", "claude-sonnet-5")   # ゲーム生成・重い修正
MODEL_POLISH  = ("anthropic", "claude-opus-5")     # 作り込み（Opus選択時）

# GEMINI_API_KEY 未設定時のフォールバック先
MODEL_FALLBACK = ("anthropic", "claude-sonnet-5")

# 作り込みで選べるモデル（画面の既定は sonnet）
REFINE_MODELS = {
    "sonnet": ("anthropic", "claude-sonnet-5"),
    "opus": MODEL_POLISH,
    "fable": ("anthropic", "claude-fable-5"),
}

# ---------- 掛け合わせ対象の抽選データ ----------
# サーバー側の抽選用。プロンプトには抽選結果の10語だけを載せ、全体は送らない。

COMBO_POOL = {
    "具体的な遊び": [
        "かくれんぼ", "しりとり", "椅子取りゲーム", "だるまさんが転んだ",
        "缶蹴り", "福笑い", "けん玉", "双六", "あやとり", "なわとび",
        "ドッジボール", "宝探し", "坊主めくり", "かるた", "風船割り",
        "ジェンガ", "神経衰弱", "腕相撲", "リレー", "じゃんけん",
    ],
    "物理・環境": [
        "重力", "摩擦", "温度", "天気", "水中", "風", "弾性", "浮力",
    ],
    "知覚・感覚": [
        "視界", "音", "匂い", "触覚", "入力の遅延", "色の意味", "距離感",
    ],
    "時間・順番": [
        "巻き戻し", "予約", "待機", "同時進行", "逆再生", "早送り",
    ],
    "社会・関係": [
        "通貨", "契約", "投票", "噂", "名前", "順位", "貸し借り",
    ],
    "抽象・概念": [
        "記憶", "嘘", "確率", "成長", "消失", "模倣", "秘密", "約束",
    ],
}

# 10案の配分（合計が10になるようにすること）
COMBO_MIX = {
    "具体的な遊び": 4,
    "物理・環境": 1,
    "知覚・感覚": 1,
    "時間・順番": 1,
    "社会・関係": 1,
    "抽象・概念": 2,
}

app = FastAPI(title="Game Idea Tool")
SAVED_DIR.mkdir(exist_ok=True)


# ---------- モデル呼び出し（プロバイダ共通） ----------

def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail="環境変数 ANTHROPIC_API_KEY が設定されていません。",
        )
    return key


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


async def call_anthropic(model: str, prompt: str, max_tokens: int, timeout: float) -> str:
    headers = {
        "x-api-key": anthropic_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Claude 5系は思考が既定でONになり、思考分も max_tokens を消費して
    # 本文が打ち切られるため、明示的にOFFにする（思考トークンは出力課金でもある）。
    # ただし Fable は思考OFF指定を受け付けない（400になる）ため指定しない。
    if "fable" not in model and "mythos" not in model:
        payload["thinking"] = {"type": "disabled"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="生成がタイムアウトしました。もう一度試してください。")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"APIエラー ({r.status_code}): {r.text[:200]}")
        data = r.json()
    usage = data.get("usage", {})
    print(f"[anthropic] model={model} in={usage.get('input_tokens')} out={usage.get('output_tokens')}")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class GeminiTransientError(Exception):
    """Gemini側の一時的な失敗（混雑・障害・タイムアウト）。Claudeへのフォールバック対象。"""


async def call_gemini(model: str, prompt: str, max_tokens: int, timeout: float) -> str:
    headers = {
        "x-goog-api-key": gemini_key(),
        "content-type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # 思考型モデルは内部思考が出力枠を使うことがあるため、下限を確保する（Flash料金では誤差）
        "generationConfig": {"maxOutputTokens": max(max_tokens, 1024)},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(GEMINI_API_URL.format(model=model), headers=headers, json=payload)
        except httpx.TimeoutException:
            raise GeminiTransientError("タイムアウト")
        if r.status_code in (429, 500, 502, 503, 504):
            raise GeminiTransientError(f"HTTP {r.status_code}: {r.text[:120]}")
        if r.status_code != 200:
            # キー不正など恒常的なエラーは隠さず表面化させる
            raise HTTPException(status_code=502, detail=f"Gemini APIエラー ({r.status_code}): {r.text[:200]}")
        data = r.json()
    usage = data.get("usageMetadata", {})
    print(f"[gemini] model={model} in={usage.get('promptTokenCount')} out={usage.get('candidatesTokenCount')}")
    candidates = data.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts") if candidates else None
    text = "".join(p.get("text", "") for p in parts or [])
    if not text.strip():
        raise HTTPException(status_code=502, detail="Geminiの応答が空でした。もう一度試してください。")
    return text


CLAUDE_CLI_PATH = shutil.which("claude")


def _run_cli(cmd: list, prompt_bytes: bytes, timeout: float):
    # ANTHROPIC_API_KEY があるとCLIがAPIキー課金で動いてしまうため、
    # 子プロセスからは除外し、ログイン済みアカウント（定額枠）で確実に動かす
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return subprocess.run(cmd, input=prompt_bytes, capture_output=True, timeout=timeout, env=env)


async def call_claude_cli(model_alias: str, prompt: str, timeout: float) -> str:
    if not CLAUDE_CLI_PATH:
        raise HTTPException(status_code=500, detail="claude CLI が見つかりません。USE_CLAUDE_CLI を False にするか、Claude Code をインストールしてください。")
    cmd = [CLAUDE_CLI_PATH, "-p", "--output-format", "text", "--model", model_alias]
    if CLAUDE_CLI_PATH.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c"] + cmd
    try:
        # CLIは起動コストがあるため、猶予を60秒足す。プロンプトはstdin渡し（コマンド長制限の回避）
        res = await asyncio.to_thread(_run_cli, cmd, prompt.encode("utf-8"), timeout + 60)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="生成がタイムアウトしました。もう一度試してください。")
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace")[:200]
        raise HTTPException(status_code=502, detail=f"Claude CLI エラー: {err}")
    text = res.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Claude CLI の応答が空でした。もう一度試してください。")
    print(f"[claude-cli] model={model_alias} out_chars={len(text)}")
    return text


async def call_model(model: tuple, prompt: str, max_tokens: int, timeout: float) -> str:
    """(プロバイダ, モデル名) を受け取り、適切なAPIを叩く。呼び出し側はプロバイダを意識しない。"""
    if USE_CLAUDE_CLI:
        # 検証モード: 定額枠のCLIで動かす。用途ごとの本来のモデルに合わせる
        # （ゲーム生成・アイデア・修正は sonnet で速く、作り込みは選んだモデル）。
        name = model[1]
        if "fable" in name:
            alias = "claude-fable-5"
        elif "opus" in name:
            alias = "opus"
        else:
            alias = "sonnet"
        return await call_claude_cli(alias, prompt, timeout)
    provider, name = model
    if provider == "gemini":
        if gemini_key():
            try:
                return await call_gemini(name, prompt, max_tokens, timeout)
            except GeminiTransientError as e:
                print(f"[fallback] Geminiが一時的に利用できないため {MODEL_FALLBACK[1]} で代替します ({e})")
                provider, name = MODEL_FALLBACK
        else:
            print(f"[fallback] GEMINI_API_KEY が未設定のため、{name} の代わりに {MODEL_FALLBACK[1]} を使います")
            provider, name = MODEL_FALLBACK
    return await call_anthropic(name, prompt, max_tokens, timeout)


def strip_html_output(text: str) -> str:
    html = text.strip()
    html = re.sub(r"^```(?:html)?\s*", "", html)
    html = re.sub(r"\s*```$", "", html)
    low = html.lower()
    if "<html" not in low:
        raise HTTPException(status_code=502, detail="ゲームの生成に失敗しました。もう一度試してください。")
    # 途中で打ち切られた不完全なHTMLを保存しないよう、閉じタグまで出力されたことを確認する
    if not low.endswith("</html>"):
        raise HTTPException(status_code=502, detail="生成が途中で打ち切られました（出力が長すぎた可能性）。もう一度試してください。")
    return html


def parse_json_output(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
        if not m:
            raise HTTPException(status_code=502, detail="応答の解析に失敗しました。もう一度試してください。")
        return json.loads(m.group(0))


# ---------- 保存 ----------

ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


def new_game_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def check_id(game_id: str) -> None:
    if not ID_RE.match(game_id):
        raise HTTPException(status_code=400, detail="不正なIDです。")


def load_meta(game_id: str) -> dict:
    check_id(game_id)
    path = SAVED_DIR / f"{game_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="保存データが見つかりません。")
    return json.loads(path.read_text(encoding="utf-8"))


def load_html(game_id: str) -> str:
    check_id(game_id)
    path = SAVED_DIR / f"{game_id}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="保存データが見つかりません。")
    return path.read_text(encoding="utf-8")


def save_meta(meta: dict) -> None:
    (SAVED_DIR / f"{meta['id']}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_game(meta: dict, html: str) -> None:
    (SAVED_DIR / f"{meta['id']}.html").write_text(html, encoding="utf-8")
    save_meta(meta)


def backup_and_write(game_id: str, new_html: str) -> None:
    """修正の適用前に直前のHTMLを退避してから上書きする（「元に戻す」用・1段階）"""
    cur = SAVED_DIR / f"{game_id}.html"
    (SAVED_DIR / f"{game_id}.prev.html").write_text(
        cur.read_text(encoding="utf-8"), encoding="utf-8"
    )
    cur.write_text(new_html, encoding="utf-8")


def can_undo(game_id: str) -> bool:
    return (SAVED_DIR / f"{game_id}.prev.html").exists()


# ---------- 本質抽出 ----------

# 依頼元の指定文言。要約・改変しないこと。
ESSENCE_PROMPT = """次の言葉について、それぞれ「プレイヤーが何を判断しているか」を一文で書いてください。

条件:
- 「何が起きるか」ではなく「何を選んでいるか」を書くこと
- 崩れる・逃げる・隠れる・座る といった現象を答えにしないこと
- その遊びから「勝ち負けの条件」を取り除いても残るものを書くこと
- 遊びでない語（抽象概念など）の場合は、それが人間の判断にどう影響するかを書くこと

対象: {words}

出力は以下のJSONのみ。前置き・コードブロック記号は書かないこと。
[{{"word": "語", "essence": "一文"}}, ...]"""


async def extract_essences(targets: list) -> dict:
    """抽選した10語をまとめて1回で渡し、語→本質のdictを返す。失敗時は空文字で続行。"""
    try:
        text = await call_model(
            MODEL_ESSENCE,
            ESSENCE_PROMPT.format(words=", ".join(targets)),
            max_tokens=1000,
            timeout=60.0,
        )
        items = parse_json_output(text)
        mapping = {}
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("word"):
                    mapping[str(it["word"]).strip()] = str(it.get("essence", "")).strip()
        return {t: mapping.get(t, "") for t in targets}
    except HTTPException as e:
        print(f"[essence] 本質抽出に失敗したため語のみで続行します: {e.detail}")
        return {t: "" for t in targets}


# ---------- アイデア生成 ----------

class IdeaRequest(BaseModel):
    theme: str = ""


IDEA_PROMPT = """あなたはゲームデザイナーです。
{theme_line}

以下の10個の対象それぞれについて、その対象を中心に据えたゲームのアイデアを1個ずつ、合計10個考えてください。
各対象には「プレイヤーが何を判断しているか」（本質）を添えています。

{target_lines}

条件:
- 各案は、割り当てられた対象を必ず中心に据えること。対象を差し替えたり無視したりしないこと
- 対象から連想される表面的な現象ではなく、添えられた本質（プレイヤーの判断）を遊びの中心に据えること
- 相性が悪く見える組み合わせでも、無理に丸く成立させようとせず、その衝突から何が生まれるかを考えること
- ブラウザ上の単一HTMLファイルで、シンプルに実装できる規模にすること
- 面白さの当たり外れは問いません。振れ幅を大きくしてください

出力は以下のJSON配列のみ。1番から10番の対象と同じ順に並べること。前置き・後書き・コードブロック記号は一切書かないこと。

[
  {{"title": "8文字程度の短い名前", "desc": "どんなゲームか2文で説明"}},
  ... 10個
]"""


def draw_targets() -> list:
    targets = []
    for category, count in COMBO_MIX.items():
        targets += random.sample(COMBO_POOL[category], count)
    random.shuffle(targets)
    return targets


@app.post("/api/ideas")
async def gen_ideas(req: IdeaRequest):
    theme = req.theme.strip()
    targets = draw_targets()

    # 前段: 本質抽出（1回の呼び出しで10語まとめて）
    essences = await extract_essences(targets)

    theme_line = (
        f"テーマの指定: 「{theme}」\n各案は「このテーマ × 対象」の掛け合わせで考えてください。"
        if theme
        else "テーマの指定はありません。各対象を軸に自由に発想してください。"
    )
    target_lines = "\n".join(
        f"{i + 1}. {t}" + (f"（本質: {essences[t]}）" if essences.get(t) else "")
        for i, t in enumerate(targets)
    )
    text = await call_model(
        MODEL_IDEAS,
        IDEA_PROMPT.format(theme_line=theme_line, target_lines=target_lines),
        max_tokens=3000,
        timeout=120.0,
    )
    raw = parse_json_output(text)

    # 対象の割り当てはサーバー側の抽選結果を正とする（モデルの出力順に対応付け）
    ideas = []
    for i, item in enumerate(raw[:10]):
        target = targets[i] if i < len(targets) else ""
        combo = f"{theme} × {target}" if theme else target
        ideas.append({
            "title": item.get("title", ""),
            "desc": item.get("desc", ""),
            "target": target,
            "combo": combo,
        })

    # 抽出結果をその回のアイデアデータと一緒に保存（後から検証・再訪するため）
    at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(IDEAS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "at": at,
                "theme": theme,
                "targets": targets,
                "essences": essences,
                "ideas": ideas,
            }, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[ideas-log] 保存に失敗: {e}")

    return {"ideas": ideas, "at": at}


@app.get("/api/ideas/history")
async def ideas_history(limit: int = 20):
    """過去に生成したアイデアの回を新しい順で返す（AIは呼ばない）"""
    if not IDEAS_LOG.exists():
        return {"rounds": []}
    rounds = []
    for line in IDEAS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rounds.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rounds.reverse()
    return {"rounds": rounds[:max(1, min(limit, 100))]}


# ---------- ゲーム生成 ----------

# 手触りの数値を1箇所に集約させる指示。新規生成・作り込みの両方に含める（依頼元の指定文言）。
CONFIG_INSTRUCTION = """手触りに関わる数値は、コード冒頭の CONFIG オブジェクトに集約してください。

const CONFIG = {
  playerSpeed: 5,      // プレイヤーの移動速度
  jumpPower: 12,       // ジャンプの高さ
  enemySpeed: 3,       // 敵の速度
  hitboxScale: 1.0,    // 当たり判定の大きさ
  // 他、このゲームで手触りに効く値
};

各項目には、日本語のコメントで「何の値か」を必ず書いてください。
CONFIG の値を変えるだけで手触りが変わるよう、
コード内では必ずこの値を参照してください。"""


# ローカライズ検証（LQA）と堅牢性のための指示。生成・作り込みの両方に含める。
LQA_INSTRUCTION = """【文字列テーブル】
プレイヤーに見えるテキストは、すべてコード冒頭の
const STRINGS = { key: "表示文字列", ... } に集約してください。
コード中では STRINGS.key の形でのみ参照し、文字列リテラルを直接書かないでください。

【ラベル描画】
ボタン・パネルなど「決まった枠の中」に文字を描くときは、
必ず drawLabel(ctx, text, x, y, w, h) を使ってください（この関数はツール側が用意します。自前で fillText を書かないでください）。
枠を持たない演出テキストは、これまでどおり fillText で構いません。

【エラーを握り潰さない】
try/catch で囲む場合、catch した例外は必ず console.error(e) に出力してください
（握り潰すと自動検証がエラーを検出できません）。"""


class GameRequest(BaseModel):
    title: str
    combo: str = ""
    desc: str = ""
    target: str = ""


def build_game_prompt(req: GameRequest) -> str:
    # CONFIG_INSTRUCTION 内の波括弧が str.format に解釈されないよう、f-string で組み立てる
    return f"""次のゲームアイデアを、遊べるデモとして実装してください。

タイトル: {req.title}
掛け合わせ: {req.combo}
内容: {req.desc}

要件:
- 単一のHTMLファイルとして完結させること（外部ライブラリの読み込みは禁止）
- <!DOCTYPE html> から </html> まで、完全な形で出力すること
- **スマートフォンのタッチ操作で遊べること。** タップ・スワイプ・長押しのいずれかで操作する
- PCのマウス・キーボードでも操作できるようにすること
- 画面サイズに合わせて自動でフィットさせること（縦向きのスマホを想定）
- Canvas または DOM で描画する。図形と色だけで構わない
- 開始と同時に遊べること。説明画面は不要だが、操作方法を画面内に短く表示すること
- スコアや進行など、目的が分かる要素を最低ひとつ入れること
- 操作に浅いコツを1つ入れること（例：タイミングよく押すと少し有利になる）。コツが決まったことが分かる表示（加点やエフェクト）を入れること
- エラーで止まらないことを最優先。凝った実装より確実に動くことを選ぶこと

{CONFIG_INSTRUCTION}

{LQA_INSTRUCTION}

HTMLコードのみを出力してください。説明文やコードブロック記号は書かないこと。"""


@app.post("/api/game")
async def gen_game(req: GameRequest):
    text = await call_model(
        MODEL_GAME,
        build_game_prompt(req),
        max_tokens=16000,
        timeout=300.0,
    )
    html = strip_html_output(text)
    meta = {
        "id": new_game_id(),
        "title": req.title,
        "combo": req.combo,
        "target": req.target,
        "desc": req.desc,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "refined": False,
    }
    save_game(meta, html)
    return {"html": html, "id": meta["id"]}


# ---------- 保存済みゲーム（外部APIは呼ばない） ----------

@app.get("/api/saved")
async def list_saved():
    games = []
    for path in SAVED_DIR.glob("*.json"):
        try:
            games.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    games.sort(key=lambda g: g.get("id", ""), reverse=True)
    return {"games": games}


@app.get("/api/saved/{game_id}")
async def get_saved(game_id: str):
    meta = load_meta(game_id)
    return {"html": load_html(game_id), "meta": meta, "can_undo": can_undo(game_id)}


@app.delete("/api/saved/{game_id}")
async def delete_saved(game_id: str):
    check_id(game_id)
    removed = False
    for suffix in (".html", ".json"):
        path = SAVED_DIR / f"{game_id}{suffix}"
        if path.exists():
            path.unlink()
            removed = True
    prev = SAVED_DIR / f"{game_id}.prev.html"
    if prev.exists():
        prev.unlink()
    if not removed:
        raise HTTPException(status_code=404, detail="保存データが見つかりません。")
    return {"ok": True}


class IssuesRequest(BaseModel):
    issues: list = []


@app.post("/api/saved/{game_id}/issues")
async def set_issues(game_id: str, req: IssuesRequest):
    """ブラウザ側の自動検証で見つかった問題を記録する（AIは呼ばない）。
    空リストで記録をクリア。記録は作り込み・構造修正のプロンプトに制約として注入される。"""
    meta = load_meta(game_id)
    meta["issues"] = [str(s)[:300] for s in req.issues][:10]
    save_meta(meta)
    return {"ok": True, "issues": meta["issues"]}


@app.post("/api/saved/{game_id}/undo")
async def undo_saved(game_id: str):
    check_id(game_id)
    cur = SAVED_DIR / f"{game_id}.html"
    prev = SAVED_DIR / f"{game_id}.prev.html"
    if not cur.exists() or not prev.exists():
        raise HTTPException(status_code=404, detail="戻せる状態がありません。")
    cur_html = cur.read_text(encoding="utf-8")
    prev_html = prev.read_text(encoding="utf-8")
    cur.write_text(prev_html, encoding="utf-8")
    prev.write_text(cur_html, encoding="utf-8")  # 入れ替え式なので、もう一度押すとやり直しになる
    return {"html": prev_html, "can_undo": True}


# ---------- 作り込み ----------

# 要求0〜2は依頼元の指定文言。要約・改変しないこと。
REFINE_REQUIREMENTS = """【要求0：元のアイデアの核心を守る】
まず、このゲームの核心を一文で特定してください。
「プレイヤーが何を判断する遊びなのか」を書いてください。

その上で、作り込みによってこの核心が薄まってはなりません。
追加・変更する要素は、すべて核心を強化するものに限ってください。
核心と無関係な一般的なゲーム要素（作戦選択、レベル、スコア倍率、
難易度設定など）を持ち込まないでください。

特定した核心を、コード冒頭のコメントに日本語で明記してください。

【要求1：選択肢を作る】
プレイヤーが繰り返し遊ぶうちに上手くなる余地を作ってください。
具体的には、「最短で成功するがリスクの高い方法」と「安全だが時間がかかる方法」の
両方が成立するようにし、プレイヤーがその場で選べるようにしてください。
難易度の数値を上げるだけの対応は避けてください。
ただし、この選択肢は要求0で特定した核心の上に成立するものにしてください。

【要求2：上級者向けのテクニックを1つ設計する】
上級者だけが使える操作テクニックを、最低1つ設計して実装してください。
そして、それが何であるかを、コード冒頭のコメントに日本語で明記してください。
（例：「特定のタイミングで入力を重ねると、通常より速く移動できる」など）
ただし、このテクニックは要求0で特定した核心を深めるものにしてください。

【要求3：操作の工夫が上達に変わるようにする】
操作に「慣れると上手くなる余地」を最低1つ組み込んでください。
（例：長押しの長さで強さが変わる、タイミングを合わせると効果が上がる、
連続入力の精度で結果が変わる、など）
その工夫が上手く決まったときは、画面上のフィードバック（エフェクトや加点表示）で
プレイヤーに伝わるようにしてください。伝わらない工夫は存在しないのと同じです。
ただし、この工夫は要求0で特定した核心を深めるものに限ってください。"""


class RefineRequest(BaseModel):
    model: str = "sonnet"


def issues_block(meta: dict) -> str:
    """検証で記録された問題があれば、再発防止の制約節を返す（層2: 検証結果の還流）"""
    issues = meta.get("issues") or []
    if not issues:
        return ""
    lines = "\n".join(f"- {i}" for i in issues)
    return f"""検証で以下の問題が確認されています:
{lines}
新しい版では、これらの問題が発生しないようにしてください。

"""


def build_refine_prompt(meta: dict, current_html: str) -> str:
    # HTML内の波括弧が str.format に解釈されないよう、f-string で組み立てる
    return f"""次のゲームを、元の骨格を保ったまま、繰り返し遊べる深さを持った版に作り直してください。

タイトル: {meta.get("title", "")}
掛け合わせ: {meta.get("combo", "")}
内容: {meta.get("desc", "")}

現在のHTML:
{current_html}

{REFINE_REQUIREMENTS}

{CONFIG_INSTRUCTION}

{LQA_INSTRUCTION}

{issues_block(meta)}その他の条件:
- 元のゲームの骨格は保つこと。別のゲームに作り変えないこと
- 単一のHTMLファイルとして完結させること（外部ライブラリの読み込みは禁止）
- <!DOCTYPE html> から </html> まで、完全な形で出力すること
- スマートフォンのタッチ操作で遊べること。PCのマウス・キーボードでも操作できるようにすること
- エラーで止まらないことを最優先。凝った実装より確実に動くことを選ぶこと

HTMLコードのみを出力してください。説明文やコードブロック記号は書かないこと。"""


@app.post("/api/refine/{game_id}")
async def refine_game(game_id: str, req: RefineRequest):
    if req.model not in REFINE_MODELS:
        raise HTTPException(status_code=400, detail="モデルは sonnet / opus / fable のいずれかを指定してください。")
    meta = load_meta(game_id)
    current_html = load_html(game_id)

    # Fable は思考が常時ONで作り直しに時間がかかるため、猶予を長めに取る
    refine_timeout = 540.0 if req.model == "fable" else 240.0
    text = await call_model(
        REFINE_MODELS[req.model],
        build_refine_prompt(meta, current_html),
        max_tokens=16000,
        timeout=refine_timeout,
    )
    html = strip_html_output(text)

    # 上書きせず、別の版として保存する（Sonnet版とOpus版を比較できるようにする）
    new_meta = {
        "id": new_game_id(),
        "title": meta.get("title", ""),
        "combo": meta.get("combo", ""),
        "target": meta.get("target", ""),
        "desc": meta.get("desc", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "refined": True,
        "refine_model": req.model,
        "parent_id": game_id,
    }
    save_game(new_meta, html)
    return {"html": html, "meta": new_meta}


# ---------- 修正依頼 ----------

class FixRequest(BaseModel):
    request: str


def find_config(html: str):
    """HTMLから `const CONFIG = {...}` の範囲を抜き出す。(開始, 終了, テキスト) または None"""
    m = re.search(r"const\s+CONFIG\s*=\s*\{", html)
    if not m:
        return None
    depth = 0
    for j in range(m.end() - 1, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return (m.start(), j + 1, html[m.start():j + 1])
    return None


# 値として置換できるのは数値・真偽値・文字列リテラルのみ（コメントは保持される）
VALUE_RE = r"""(-?\d+\.?\d*|true|false|"[^"]*"|'[^']*')"""


def apply_config_changes(config_text: str, changes: dict):
    """CONFIG テキスト内の該当キーの値だけを置き換える。(新テキスト, 適用結果dict)"""
    applied = {}
    for key, value in changes.items():
        if not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", str(key)):
            continue
        pattern = rf"(\b{re.escape(key)}\s*:\s*){VALUE_RE}"
        new_value = json.dumps(value, ensure_ascii=False)
        new_text, n = re.subn(pattern, lambda mm: mm.group(1) + new_value, config_text, count=1)
        if n:
            config_text = new_text
            applied[key] = value
    return config_text, applied


def build_judge_prompt(request_text: str, config_text: str) -> str:
    return f"""あなたはゲームの修正依頼を振り分ける係です。

修正依頼: 「{request_text}」

このゲームで調整できる設定値（CONFIG）:
{config_text}

この依頼が、上記の設定値の変更だけで対応できるなら "light"、
コードの構造変更（ルールの追加・削除、新しい要素、見た目や表示の変更など）が必要なら "heavy" と判定してください。

出力は以下のJSONのみ。他は一切書かないこと。
{{"type": "light" または "heavy", "reason": "判定理由を1文で"}}"""


def build_tune_prompt(request_text: str, config_text: str) -> str:
    return f"""ゲームの設定値を、修正依頼に合わせて調整してください。

修正依頼: 「{request_text}」

現在の設定値（CONFIG）:
{config_text}

変更が必要な項目だけを、以下のJSON形式で返してください。JSON以外は一切書かないこと。
{{"項目名": 新しい値, ...}}"""


def build_heavy_fix_prompt(request_text: str, current_html: str, meta: dict) -> str:
    return f"""次のゲームを、修正依頼に沿って修正してください。

修正依頼: 「{request_text}」

現在のHTML:
{current_html}

{issues_block(meta)}条件:
- 修正依頼に関わる部分以外は、できるだけ変えないこと。別のゲームに作り変えないこと
- 単一のHTMLファイルとして完結させること（外部ライブラリの読み込みは禁止）
- <!DOCTYPE html> から </html> まで、完全な形で出力すること
- 手触りに関わる数値は、引き続きコード冒頭の CONFIG オブジェクトに集約すること
- スマートフォンのタッチ操作で遊べること。PCのマウス・キーボードでも操作できるようにすること
- エラーで止まらないことを最優先。凝った実装より確実に動くことを選ぶこと

HTMLコードのみを出力してください。説明文やコードブロック記号は書かないこと。"""


@app.post("/api/fix/{game_id}")
async def fix_game(game_id: str, req: FixRequest):
    """修正依頼の受付。軽い修正（数値調整）ならその場で適用し、構造変更なら確認待ちで返す。"""
    request_text = req.request.strip()
    if not request_text:
        raise HTTPException(status_code=400, detail="修正内容を入力してください。")
    load_meta(game_id)
    html = load_html(game_id)

    cfg = find_config(html)
    if cfg is None:
        # 古いゲームなどCONFIGが無い場合は、数値調整ができないので構造変更として扱う
        return {
            "type": "heavy",
            "reason": "このゲームには設定値（CONFIG）が無いため、全体の作り直しが必要です。",
            "can_undo": can_undo(game_id),
        }
    start, end, config_text = cfg

    # 1) 振り分け判定（CONFIGと要望だけを送る。HTML全体は送らない）
    judge_text = await call_model(
        MODEL_TRIAGE,
        build_judge_prompt(request_text, config_text),
        max_tokens=200,
        timeout=60.0,
    )
    judge = parse_json_output(judge_text)
    if judge.get("type") == "heavy":
        return {
            "type": "heavy",
            "reason": judge.get("reason", "構造の変更が必要と判定されました。"),
            "can_undo": can_undo(game_id),
        }

    # 2) 軽い修正: 新しい値だけをJSONで受け取り、サーバー側で置換する
    tune_text = await call_model(
        MODEL_CONFIG,
        build_tune_prompt(request_text, config_text),
        max_tokens=1000,
        timeout=60.0,
    )
    changes = parse_json_output(tune_text)
    if not isinstance(changes, dict):
        raise HTTPException(status_code=502, detail="調整結果の解析に失敗しました。もう一度試してください。")

    new_config, applied = apply_config_changes(config_text, changes)
    if not applied:
        return {
            "type": "light",
            "reason": judge.get("reason", ""),
            "applied": {},
            "html": html,
            "can_undo": can_undo(game_id),
        }

    new_html = html[:start] + new_config + html[end:]
    backup_and_write(game_id, new_html)
    return {
        "type": "light",
        "reason": judge.get("reason", ""),
        "applied": applied,
        "html": new_html,
        "can_undo": True,
    }


@app.post("/api/fix/{game_id}/heavy")
async def fix_game_heavy(game_id: str, req: FixRequest):
    """構造の変更（利用者がコスト確認済みの場合のみ呼ばれる）。HTML全体を渡して作り直す。"""
    request_text = req.request.strip()
    if not request_text:
        raise HTTPException(status_code=400, detail="修正内容を入力してください。")
    meta = load_meta(game_id)
    html = load_html(game_id)

    text = await call_model(
        MODEL_GAME,
        build_heavy_fix_prompt(request_text, html, meta),
        max_tokens=16000,
        timeout=300.0,
    )
    new_html = strip_html_output(text)
    backup_and_write(game_id, new_html)
    return {"type": "heavy_done", "html": new_html, "can_undo": True}


# ---------- エラー修復（層3: 自動検証で見つかったエラーの修正） ----------

class DebugFixRequest(BaseModel):
    errors: list = []


def build_debug_fix_prompt(errors: list, current_html: str) -> str:
    lines = "\n".join(f"- {e}" for e in errors)
    return f"""次のゲームの実行中に、以下のエラーが検出されました。

検出されたエラー:
{lines}

現在のHTML:
{current_html}

条件:
- エラーの原因を特定して修正すること
- ゲームの内容・ルール・挙動は変えないこと
- 単一のHTMLファイルとして完結させること（外部ライブラリの読み込みは禁止）
- <!DOCTYPE html> から </html> まで、完全な形で出力すること
- 手触りに関わる数値は、引き続きコード冒頭の CONFIG オブジェクトに集約すること

HTMLコードのみを出力してください。説明文やコードブロック記号は書かないこと。"""


@app.post("/api/fix/{game_id}/debug")
async def fix_game_debug(game_id: str, req: DebugFixRequest):
    errors = [str(e)[:300] for e in req.errors][:10]
    if not errors:
        raise HTTPException(status_code=400, detail="エラー内容がありません。")
    load_meta(game_id)
    html = load_html(game_id)

    text = await call_model(
        MODEL_GAME,
        build_debug_fix_prompt(errors, html),
        max_tokens=16000,
        timeout=300.0,
    )
    new_html = strip_html_output(text)
    backup_and_write(game_id, new_html)
    return {"type": "debug_done", "html": new_html, "can_undo": True}


# ---------- ローカライズ検証（LQA）の修復 ----------

class LocaleFixRequest(BaseModel):
    issues: list = []   # [{label, text, need, box, over, locale}]


def build_locale_fix_prompt(issues: list, current_html: str) -> str:
    lines = []
    for it in issues:
        lines.append(
            f"- 対象『{it.get('label', '')}』：必要幅 {it.get('need')}px / 枠幅 {it.get('box')}px"
            f"（{it.get('over')}px はみ出し／例: {it.get('locale', '')}『{it.get('text', '')}』）"
        )
    body = "\n".join(lines)
    return f"""次のゲームを別の言語で表示すると、以下の要素が枠に入りきらず表示が壊れます。

表示不具合:
{body}

現在のHTML:
{current_html}

条件:
- 枠のサイズを広げるか、フォントサイズを下げることで、長い翻訳でも収まるようにすること
- 訳文（STRINGS の文言）自体は変更しないこと。ゲームの挙動・ルールも変更しないこと
- 通常時（日本語）の見た目が大きく崩れないようにすること
- 単一のHTMLファイルとして完結させること（外部ライブラリの読み込みは禁止）
- <!DOCTYPE html> から </html> まで、完全な形で出力すること

HTMLコードのみを出力してください。説明文やコードブロック記号は書かないこと。"""


@app.post("/api/fix/{game_id}/locale")
async def fix_game_locale(game_id: str, req: LocaleFixRequest):
    issues = req.issues[:20]
    if not issues:
        raise HTTPException(status_code=400, detail="修正対象の不具合がありません。")
    load_meta(game_id)
    html = load_html(game_id)

    text = await call_model(
        MODEL_GAME,
        build_locale_fix_prompt(issues, html),
        max_tokens=16000,
        timeout=300.0,
    )
    new_html = strip_html_output(text)
    backup_and_write(game_id, new_html)
    return {"type": "locale_done", "html": new_html, "can_undo": True}


# ---------- 静的ファイル ----------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n  http://localhost:8000  をブラウザで開いてください")
    print("  スマホからは  http://<PCのIPアドレス>:8000  でアクセスできます\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
