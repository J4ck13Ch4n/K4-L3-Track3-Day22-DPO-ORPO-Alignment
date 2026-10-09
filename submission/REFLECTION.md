# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trần Hữu Đức (2A202602459)
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.
> **Nguồn dữ liệu:** NB0–NB3b chạy ở lần Kaggle thứ nhất (`colab/lab22-vinuni.ipynb`); NB4–NB7 chạy lại ở lần thứ hai sau khi
> sửa lỗi hết VRAM (`colab/lab22-kaggle-run2.ipynb`). Phiên Kaggle bị mất trước khi tải hết file, nên
> `benchmark_results.json`, `grpo_metrics.json`, `deploy_meta.json` được dựng lại từ số in ra trong notebook (có ghi `_note` trong file).
> `adapters/sft-mini/adapter_config.json` cũng được **dựng lại** (không có trọng số): các siêu tham số LoRA (r=16, alpha=32, 7 module đích, dropout 0) lấy
> từ `lab22/config.py` và trùng với `adapters/dpo/adapter_config.json`, chỉ đổi `base_model_name_or_path` thành mô hình gốc của NB1.
> Bằng chứng NB1 thật là notebook `colab/lab22-vinuni.ipynb` (loss SFT cuối 1,3603, đã lưu SFT adapter và bản gộp 16-bit).

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle Tesla T4 (14,56 GB), dùng 1 GPU (`CUDA_VISIBLE_DEVICES=0`) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch (125 bước, loss cuối 1,3603) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% cặp (median 94 token so với 86) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, LoRA r=16 trên SFT đã gộp) |
| Giám khảo | Hội đồng 2 reward model chạy local: Skywork-Reward-V2-Qwen3-4B (sanity 100%) + Skywork-Reward-V2-Llama-3.2-3B (sanity 100%) |
| Chi phí | 0 đồng (Kaggle miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ≈ 24 phút (thanh tiến trình 100/100 ở 24:14) |
| VRAM cao nhất | không đo riêng (T4 14,56 GB, huấn luyện không bị hết bộ nhớ) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,097 (chosen +0,428; rejected +0,331) |
| Độ chính xác reward trên held-out | 0,71 |
| Margin trên held-out | +0,089 (chosen +0,444; rejected +0,354) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 601 → 612 ký tự (held-out: 605 → 617) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên cả tập huấn luyện lẫn held-out, `rewards/chosen` và `rewards/rejected` đều **tăng** so với mốc 0 ban đầu
(train: +0,428 và +0,331; held-out: +0,444 và +0,354), và chosen tăng nhiều hơn rejected nên margin dương
(+0,097 trên train, +0,089 trên held-out). Vì chosen tăng chứ không giảm nên đây là kiểu **INTENDED**, không phải dịch chuyển
xác suất (likelihood displacement): chưa thấy trường hợp margin tăng do chosen tụt. Held-out đi cùng chiều với train và
khoảng cách giữa hai tập nhỏ (0,089 so với 0,097), nên chưa có dấu hiệu học thuộc. Chẩn đoán tự động khớp với điều tôi
thấy trên biểu đồ. Tuy nhiên tín hiệu **rất yếu**: margin chỉ ≈ 0,09 và độ chính xác reward held-out 0,71 trên 100
cặp chỉ hơn ngẫu nhiên khoảng 0,21; reward của cả chosen và rejected cùng tăng nghĩa là mô hình ưu ái cả hai kiểu
câu trả lời hơn tham chiếu SFT, chưa phân biệt rõ. Với lr 5e-6, 100 bước và 800 cặp, DPO hầu như không dịch chuyển mô hình,
điều này giải thích vì sao ở §4 hai mô hình gần như không khác nhau.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json` (hội đồng 2 RM; DPO chỉ thắng một cặp khi cả hai RM đồng ý):

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 4 | 4 | 42 | 0,50 (0,45–0,56) | 0,489 | 62,5% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0,375 (0,125–0,50) | 0,375 | 100% |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0,50 (0,50–0,50) | 0,50 | — |

Giám khảo: rm-panel (Qwen3-4B + Llama-3.2-3B) · sanity accuracy: 100% / 100% · `score_length_spearman`: −0,092 (Qwen3), −0,041 (Llama) ·
độ đồng ý giữa hai RM: 0,83 (n=58).

Khoảng tin cậy 0,45–0,56 chứa 0,5 nên **không phát hiện khác biệt** giữa DPO và SFT; 42/50 cặp hoà vì hai RM
không đồng ý hoặc điểm gần nhau. `per_judge` cho thấy hai RM lệch nhau: Qwen3 cho DPO thắng 10, thua 6 (win rate 0,54), còn
Llama cho DPO thắng 6, thua 10 (0,46). Qwen3 cùng họ với mô hình đang học và với mô hình sinh dữ liệu (Sailor2 gốc Qwen2.5), nên tôi
nghi có chút rò rỉ sở thích (preference leakage), nhưng chênh lệch nhỏ và khoảng tin cậy của từng RM đều chứa 0,5 nên tôi không kết luận mạnh.
Cả hai đều cùng nhóm Skywork với RM gán nhãn dữ liệu nên chưa độc lập hoàn toàn. Độ dài không phải nguyên nhân rõ: trên cặp dài gần bằng
nhau win rate vẫn ≈ 0,49. Ví dụ hữu ích: câu so sánh Python/JavaScript, hai mô hình trả lời gần như giống nhau nên không tạo khác biệt.
Ví dụ an toàn: câu hỏi tự hại (s4), cả SFT và DPO đều từ chối và khuyên gặp chuyên gia tâm lý với nội dung gần như trùng. Cả hai câu
trả lời còn bị dính chuỗi `<tool_call>` ở đầu, lỗi định dạng của mô hình SFT chứ không phải của DPO.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy. Giả thuyết: β nhỏ (0,05) cho mô hình lệch khỏi tham chiếu nhiều hơn nên margin held-out lớn hơn nhưng dễ dịch chuyển xác suất
và tăng độ dài; β lớn (0,5) giữ gần SFT, margin nhỏ và khó phân biệt với SFT hơn nữa. Với cấu hình hiện tại (DPO đã rất yếu ở β=0,1), tôi dự đoán
β=0,05 mới là hướng tạo ra khác biệt nhìn thấy được.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là **chạy bước sinh câu trả lời của NB4 trong một tiến trình con thay vì nạp model ngay trong kernel, và nạp reward model ở 16-bit**.
Lần chạy đầu, hai mô hình sinh câu trả lời (`del model` + `cleanup`) vẫn để lại khoảng 7 GB trên GPU. Reward model Qwen3-4B (cần khoảng 7,4 GiB ở
fp16) không còn chỗ, nên tôi thêm cơ chế tự chuyển sang 4-bit khi hết VRAM. Cách này tránh được lỗi OOM nhưng làm RM Qwen3 đạt sanity 0% (0/12 cặp), bị loại khỏi
hội đồng, và chỉ còn một giám khảo (Llama) với kết luận yếu hơn (win rate DPO 0,46). Phương án thay thế là chấp nhận một giám khảo hoặc dùng giám khảo API.
Tôi chọn sửa gốc: tách bước sinh ra tiến trình con để GPU được trả hoàn toàn khi tiến trình kết thúc. Kết quả xác nhận: lần chạy lại cả hai RM nạp 16-bit,
sanity đều 100%, độ đồng ý 0,83, và win rate DPO chuyển thành 0,50 (0,45–0,56), tức nhận xét "không phát hiện khác biệt" chắc chắn hơn. Điều làm tôi bất ngờ là việc
đo lường sai (RM 4-bit hỏng) có thể âm thầm đổi kết luận, và `sanity` chính là lưới an toàn giúp phát hiện. Làm lại, tôi sẽ đo VRAM sau mỗi bước ngay từ đầu, và
tăng lr hoặc số bước DPO vì hiện tại DPO gần như không thay đổi hành vi.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 200 | 0,530 ± 0,035 | 0,520 ± 0,035 | −0,010 |
| GSM8K | 250 | 0,828 ± 0,024 | 0,832 ± 0,024 | +0,004 |
| Global-MMLU-vi | 10 / môn | 0,244 ± 0,018 | 0,244 ± 0,018 | 0,000 |

Không có Δ nào gần 2× stderr: IFEval giảm 0,010 trong khi stderr ≈ 0,035, GSM8K tăng 0,004 với stderr ≈ 0,024, Global-MMLU-vi không đổi. Vì vậy
tôi không thấy "thuế căn chỉnh" (alignment tax): điểm GSM8K không giảm sau DPO. Điều này khớp với NB4, nơi DPO và SFT cũng không khác nhau rõ rệt, và với §3
(DPO dịch chuyển mô hình rất ít). Điểm Global-MMLU-vi 0,244 xấp xỉ mức đoán ngẫu nhiên của câu hỏi 4 đáp án (0,25) và hai điều kiện giống hệt nhau tới
chữ số thập phân thứ năm; với chỉ 10 câu mỗi môn tôi nghi bộ đo này chưa đủ lớn hoặc mô hình chưa làm tốt đề tiếng Việt, nên không nên dùng nó để kết luận về DPO.
Độ chính xác GSM8K cao (0,83) so với IFEval (0,53) cho thấy mô hình mạnh về suy luận số học với 5-shot hơn là tuân thủ ràng buộc định dạng.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0,64 | +0,025 (chosen 0,095 / rejected 0,070) | 392,0 | INTENDED |
| RPO | 0,64 | +0,035 (0,572 / 0,538) | 402,0 | INTENDED, reward tăng mạnh nhờ thêm NLL |
| DPO-norm | 0,62 | +0,010 (−0,181 / −0,190) | 400,7 | LIKELIHOOD DISPLACEMENT |
| LD-DPO | 0,57 | +0,022 (−0,128 / −0,150) | 385,0 | LIKELIHOOD DISPLACEMENT |
| ORPO | 0,65 | — (không dùng reference; log odds ratio −0,624) | 372,1 | không có chẩn đoán |

ORPO làm câu trả lời **ngắn nhất** (372 ký tự) còn RPO dài nhất (402). RPO thêm số hạng NLL trên câu chosen nên phạt việc xác suất của chosen giảm, nhờ vậy giữ chosen
tăng và có xu hướng sinh câu dài hơn. ORPO không dùng mô hình tham chiếu và bổ sung số hạng odds-ratio trên nền SFT loss, nên không bị
neo vào độ dài của tham chiếu như DPO. DPO-norm và LD-DPO chuẩn hoá theo độ dài và giảm tác động của token thừa nên đều rơi vào dịch chuyển xác suất ở cấu hình này.
Tất cả khác biệt đều nhỏ (khoảng 30 ký tự, độ chính xác 0,57–0,65 trên 100 cặp), cần thận trọng khi diễn giải.

---

## 9. GRPO (bonus NB7)

> Ảnh: `screenshots/08-grpo-reward.png`

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 0,490 / 0,500 (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ≈ 0,050 |

Tập `vuongtsc/vi-gsm8k-agentic` (400 train / 100 test), 60 bước, 4 lượt sinh mỗi câu. Độ chính xác tăng 0,01, nhỏ hơn nhiều so với sai số chuẩn ≈ 0,05, nên chênh lệch
chỉ là nhiễu. Với 60 bước (0,15 epoch) thì chưa đủ để GRPO cải thiện đáng kể. Reward định dạng (`Đáp số:`) thường được học trước reward đúng đáp án,
nhưng ở lần chạy này tôi chưa khai thác được thành phần nào vì quá ít bước.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4) — Q4_K_M 2,50 GB; câu trả lời HF và GGUF cùng nội dung (`data/eval/deploy_meta.json`)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4) — chỉ có hai RM Skywork, chưa có giám khảo API
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Đầu ra của SFT, DPO và GGUF đều bắt đầu bằng `<tool_call>` hoặc `</tool_call>` lặp lại dù prompt không liên quan đến công cụ. Tôi đã điều tra bằng tokenizer thật
(`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) và xác nhận được **một lệch định dạng giữa huấn luyện và sinh**: chat template luôn viết khối
`<think>\n\n</think>\n\n` rỗng vào lượt `assistant` của văn bản huấn luyện, nhưng generation prompt chỉ kết thúc bằng `<|im_start|>assistant\n`.
Vì `train_on_responses_only` chỉ che phần trước `<|im_start|>assistant\n`, khối think rỗng nằm trong phần bị tính loss, nên mô hình SFT học cách tự sinh nó
ở đầu mỗi câu trả lời, và cả SFT, DPO lẫn GGUF kế thừa hành vi này. Chuỗi lặp dạng `X\n\nX\n\n<câu trả lời>` trong đầu ra có đúng cấu trúc đó.
Điều chưa kiểm chứng được (không có GPU để chạy lại): vì sao các token này hiện thành `<tool_call>` thay vì `<think>`. Giả thuyết của tôi là Instruct-2507 chưa từng
được huấn luyện với `<think>`/`</think>` nên mô hình chọn token gần nghĩa nhất đã được học, nhưng đây mới là giả thuyết. Tôi đã sửa mã (`lab22/modeling.py::chat_text`
và NB5) để generation prompt cũng chứa khối think rỗng, khớp định dạng huấn luyện; bài nộp này **chưa chạy lại** sau sửa. Các số liệu NB4/NB6/NB7 ở trên được sinh với prompt cũ, nên
cần đọc chúng với lưu ý này (hai mô hình đều chịu cùng lệch nên so sánh SFT với DPO vẫn công bằng nhưng tuyệt đối thì có thể thấp hơn thực tế).
