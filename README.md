# laya-mbti

X(Twitter)アーカイブのツイートを [Laya](https://github.com/NandhaKishorM/laya) の型付き判定に通し、
MBTI の4指標（E/I・S/N・T/F・J/P）を推定する **完全クライアントサイド** の静的ウェブアプリです。
処理はすべてブラウザ内で完結し、ツイート本文・アカウント情報・結果を外部へ送信しません。

参考: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) / [X-CheckPoint](https://x-checkpoint.activetk.jp)

## 特徴

- **ZIPストリーミング解析** — `@zip.js/zip.js` を Web Worker で実行。複数ZIP・1GB超・Zip64 に対応し、メディアは読み飛ばして `data/tweets*.js` 等だけを展開します。
- **端末内推論（WebGPU優先・CPUフォールバック）** — 公式 `laya-ts` ランタイム + ONNX Runtime Web の WebGPU 実行プロバイダ。モデルは fp16 で書き出し、推論は Worker 上で動きます。WebGPU が使えない環境や実行時に失敗した場合は、CPU（WASM）で最初から推論し直します。
- **4問の2択 choice で各軸を評価** — 各軸を「外向 / 内向」のような2択で判定し、確率差（マージン）で集計します。三択は確率が薄まり、yes/no 2問はパス数が倍になるため、較正実験を経て2択1問/軸に落ち着きました。
- **判定不能を弾く** — 短文・URLのみ等は事前フィルタで除外し、肯定と否定の確率差が小さい投稿は「保留」として集計から除外します。
- **推論中の暫定表示** — 解析中も途中経過のMBTIをリアルタイムに表示します。
- **バッチ推論** — 4つの質問を1つのバッチ（行）にまとめ、複数ツイートを最大64行/パスで処理します。公式実装の `predictBatch` により、質問ごとに個別forwardしていた旧実装より大幅に速くなります。
- **Service Workerキャッシュ** — 約694MBのモデルとORTのWasmをCache Storageに永続化し、再読み込み時の再ダウンロードを防ぎます（https / localhost のみ）。
- **件数上限と均等サンプリング** — CPU推論は約1秒/パスなので、既定500件を期間全体から均等抽出して解析します（0で全件）。
- **対象期間の選択** — 全期間／直近1年／3か月／1か月／カスタム範囲。

## 使い方

1. X の「設定 → アカウント → データのアーカイブをダウンロード」でアーカイブを取得し、ZIP を用意します（複数分割されていてもまとめて選択できます）。
2. アプリに ZIP をドラッグ&ドロップします。
3. 初回のみ約694MBのモデルをダウンロードします（2回目以降はService Workerキャッシュ）。
4. 対象期間とフィルタを調整し、「MBTIを推定する」を実行します。

> 本アプリの結果は投稿文の言語的傾向を集計した非公式な推定であり、診断ではありません。X Corp. とは関係ありません。

## 開発

必要環境: Node.js 22+ / pnpm / （モデル生成時のみ）[uv](https://docs.astral.sh/uv/)

```sh
pnpm install
pnpm dev            # http://127.0.0.1:5173
pnpm dev:https      # https://<LAN IP>:5173（他端末からWebGPUで使う場合）
```

> **WebGPU はセキュアコンテキスト（https または localhost）でのみ有効です。**
> LAN の他端末から `http://192.168.x.x` のように IP 直打ちすると WebGPU が無効になり、
> アプリは CPU（WASM）で推論します（かなり時間がかかります）。自己署名HTTPSの
> `pnpm dev:https` を使うと WebGPU で推論できます（初回は証明書警告を許可）。
> `pnpm dev:lan` は http のため WebGPU は使えません。

モデル資産は `public/models/laya/` に生成済みならそのまま使われます。無い場合のみ再生成してください。
本番同様に自ホストする場合:

```sh
pnpm model:export   # public/models/laya を生成（約694MB, fp16）
pnpm build:pages    # ビルド + チェックサム/サイズ検証
pnpm preview
```

| コマンド | 内容 |
| --- | --- |
| `pnpm dev` | 開発サーバ（localhost / http） |
| `pnpm dev:https` | 自己署名HTTPSの開発サーバ（LANからWebGPUを使う場合） |
| `pnpm typecheck` | 型チェック |
| `pnpm test` | 純粋ロジックのユニットテスト（Vitest） |
| `pnpm build` | 本番ビルド |
| `pnpm build:pages` | ビルド + `validate-pages.mjs`（整合性と 1GB 予算） |
| `pnpm model:export` | 固定リビジョンのモデルを fp16 の split ONNX へ書き出し |

## 構成

```
src/lib/        # 純粋ロジック（parse / filter / aggregate / mbti / format）
src/workers/    # archive.worker（解凍・解析）, inference.worker（Laya 推論）
src/components/ # DropZone / PeriodPicker / ResultCard
scripts/        # モデル書き出し（laya-web から Apache-2.0 で vendor）と検証
.kiro/          # Kiro の steering / specs / hooks / agents / skills / powers
```

判定ロジックの詳細は [`.kiro/specs/mbti-analysis/design.md`](.kiro/specs/mbti-analysis/design.md) を参照してください。

## ライセンス

Apache-2.0。上流の帰属表示は [NOTICE](NOTICE) を参照してください。
`src/vendor/laya-ts/` と `scripts/export_laya_ts.py` は
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)（Apache-2.0）から vendoring・改変したものです。
