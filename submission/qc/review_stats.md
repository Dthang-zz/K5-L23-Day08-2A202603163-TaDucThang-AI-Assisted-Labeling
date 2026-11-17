# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 628; đối tượng sau duyệt: 592; ngưỡng accepted: IoU >= 0.9
- Đã **loại 24 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)

| Trạng thái            | Số  | Tỉ lệ           | Mẫu số                  |
| --------------------- | --- | --------------- | ----------------------- |
| accepted (giữ nguyên) | 562 | 562/628 = 89.5% | 628 box máy             |
| edited (sửa)          | 15  | 15/628 = 2.4%   | 628 box máy             |
| deleted (xoá)         | 46  | 46/628 = 7.3%   | 628 box máy             |
| added (vẽ thêm)       | 15  | 15/592 = 2.5%   | 592 đối tượng sau duyệt |

Trong số deleted + added, **2 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 13.

### Theo lớp

| Lớp           | accepted | edited | deleted | added |
| ------------- | -------- | ------ | ------- | ----- |
| person        | 354      | 3      | 15      | 3     |
| bicycle       | 6        | 1      | 3       | 0     |
| motorbike     | 131      | 4      | 2       | 6     |
| car           | 49       | 0      | 4       | 1     |
| bus           | 3        | 0      | 0       | 1     |
| truck         | 4        | 0      | 2       | 0     |
| three_wheeler | 15       | 7      | 20      | 4     |

## 2. So với gold (nhãn gán tay từ đầu)

Ảnh gold: 4; đối tượng gold: 62 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 54 | 28 | 8 | 65.9% (KTC 95%: 55–75%) | 87.1% (KTC 95%: 77–93%) |

_Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn._

(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy

- Bắt được 2/12 = 16.7% (KTC 95%: 5–45%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 0/3
  - loose: 0/3
  - extra: 1/3
  - wrong_class: 1/3
  - lọt: img_024.jpg (missing), img_033.jpg (loose), img_029.jpg (loose), img_027.jpg (extra), img_013.jpg (loose), img_023.jpg (wrong_class), img_018.jpg (wrong_class), img_009.jpg (extra), img_026.jpg (missing), img_028.jpg (missing)

## Gợi ý đọc số

- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
