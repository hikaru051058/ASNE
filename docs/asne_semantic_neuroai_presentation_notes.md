# ASNE Semantic NeuroAI Presentation Notes

Use this as a loose speaking track for `docs/asne_semantic_neuroai_presentation.html`.

## English Notes

## 1. Title

ASNE stands for Artificial Semantic Neural Evaluation. The short version is that it compares predicted cortical response signatures across controlled semantic stimuli. It is a tool layer around TRIBE v2, not a new brain model by itself.

## 2. Hypothesis

The working hypothesis is that controlled semantic differences should produce distinguishable predicted response patterns in a frozen neural encoding model. In practice, the project asks whether held-out examples land closer to the expected contrast category.

## 3. Linguistics angle

The design is close to minimal-pair thinking. Instead of changing everything about a stimulus, the contrast tries to isolate a semantic relation such as expectation, causal validity, or contradiction.

## 4. What ASNE does

ASNE turns text stimuli into model inputs, runs TRIBE v2, receives predicted cortical response arrays, aggregates those responses, compares categories, and generates reports. The core contribution is the repeatable experiment workflow.

## 5. Comparison logic

The important formula is: traditional encoding is `response = f(stimulus)`. The broader ASNE idea adds state or contrast conditions. The current most reliable version compares response signatures for controlled semantic categories.

## 6. Code workflow

At a high level: dictionaries define stimuli, the adapter calls TRIBE, evaluation computes rankings and failures, ROI/parcellation code makes the output more interpretable, and scripts generate reports. The adapter design keeps the scoring logic separate from the model runner.

## 7. Tested contrasts

The tested contrasts include expected vs. unexpected, valid vs. invalid cause-effect, contradiction vs. consistency, and approach vs. static. The project learned that some contrasts work better in text/TTS than others.

## 8. Outcome

The strongest current result is expected vs. unexpected: parcel top-1 accuracy reached 0.90 on a ten-example held-out set. Cause-effect validity is also viable at 0.80 parcel top-1. Contradiction weakened after expansion, which is useful because it shows the workflow can reject a weak signal.

## 9. Narrowing

The project began with a broader steering idea. The current result is narrower and cleaner: ASNE is strongest as a controlled semantic contrast evaluation workflow. Steering remains interesting, but the semantic benchmark layer should be strengthened first.

## 10. Boundaries

The output is predicted response space, not measured human brain activity. It is not emotion detection, diagnosis, or mind reading. The current benchmark is small, so the result should be framed as a stability check and research direction.

## 11. Close

The key point for discussion is that semantic hypotheses can be turned into controlled stimuli, model-predicted response signatures, and inspectable errors. That makes ASNE useful as a research tool and as an example of LLM-assisted research infrastructure.

## 日本語メモ

## 1. タイトル

ASNEはArtificial Semantic Neural Evaluationの略です。短く言うと、制御された意味刺激に対して予測された皮質応答シグネチャを比較する仕組みです。TRIBE v2の周りにあるツールレイヤーであり、それ自体が新しい脳モデルというわけではありません。

## 2. 仮説

作業仮説は、制御された意味差が、凍結された神経エンコーディングモデルの予測応答パターンに区別可能な違いを生むはずだ、というものです。実際には、未学習の評価例が期待される対立カテゴリに近づくかを見ています。

## 3. 意味対立との接点

設計はミニマルペア的な考え方に近いです。刺激全体を大きく変えるのではなく、期待、因果の妥当性、矛盾といった意味関係をできるだけ切り出して操作します。

## 4. ASNEがすること

ASNEはテキスト刺激をモデル入力にし、TRIBE v2を実行し、予測皮質応答配列を受け取り、それを集約し、カテゴリを比較して、レポートを生成します。中心的な貢献は、再現可能な実験ワークフローです。

## 5. 比較ロジック

重要な式は、従来のエンコーディングが `response = f(stimulus)` だという点です。ASNEの広いアイデアでは、そこに状態や対立条件を加えます。現時点で最も信頼できる形は、制御された意味カテゴリの応答シグネチャを比較することです。

## 6. コードの流れ

大まかには、辞書が刺激を定義し、アダプタがTRIBEを呼び、評価コードがランキングと失敗例を計算し、ROI/パーセレーションのコードが出力を解釈しやすくし、スクリプトがレポートを生成します。アダプタ設計により、スコアリングロジックとモデル実行部を分けています。

## 7. テストした対立

テストした対立には、期待通り vs. 予想外、妥当な因果 vs. 不自然な因果、矛盾 vs. 一貫、接近 vs. 静止があります。このプロジェクトから、テキスト/TTSでうまく働く対立とそうでない対立が見えてきました。

## 8. 結果

現在最も強い結果は expected vs. unexpected で、10件の未学習セットに対してパーセルtop-1精度が0.90でした。cause-effect validity も有望で、パーセルtop-1が0.80でした。一方で contradiction は評価を拡大すると弱くなりました。これは、弱い信号を棄却できるワークフローになっているという意味でも有用です。

## 9. 焦点化

プロジェクトは、より広いステアリングのアイデアから始まりました。現在の結果はより狭く、より明確です。ASNEは今のところ、制御された意味対立評価ワークフローとして最も強いです。ステアリングはまだ面白い方向ですが、まず意味ベンチマーク層を強くするべきです。

## 10. 境界

出力は予測応答空間であり、実測されたヒト脳活動ではありません。感情検出、診断、読心ではありません。現在のベンチマークは小規模なので、結果は安定性チェックと研究方向の判断材料として扱うべきです。

## 11. 締め

議論の核は、意味的仮説を、制御刺激、モデル予測応答シグネチャ、検査可能な失敗例に変換できる点です。そのためASNEは研究ツールとしても、LLM支援による研究インフラ構築の例としても見せる価値があります。
