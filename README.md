# Bank Marketing: พยากรณ์ลูกค้าที่จะฝากประจำ (Term Deposit)

โมเดล binary classification ที่ทำนายว่าลูกค้าคนไหนมีแนวโน้มจะ "ฝากประจำ" (`deposit` = yes/no) จากข้อมูลก่อน/ระหว่างแคมเปญการตลาดทางโทรศัพท์ของธนาคาร โดยเน้นออกแบบให้ผลลัพธ์ **ไม่หลอกตัวเอง** — ตรวจสอบ data leakage และ time leakage อย่างละเอียดก่อนเชื่อตัวเลข

## ไฟล์ในโปรเจกต์

| ไฟล์ | รายละเอียด |
|---|---|
| `BankMarketing.ipynb` | โน้ตบุ๊กหลัก: EDA → feature engineering → train/test split → เปรียบเทียบโมเดล → tuning → สรุปผล |
| `BankMarketing_Presentation.pptx` | สไลด์สรุปผลงานทั้งหมด 18 หน้า สำหรับนำเสนอ |
| `datascience.csv` | ข้อมูลดิบ (Bank Marketing dataset, 45,211 แถว / 17 คอลัมน์) |

## แนวทางการทำงาน

1. **สำรวจข้อมูล (EDA)** — ตรวจสอบความไม่สมดุลของคลาส (deposit=yes แค่ ~12%) และค่า `unknown` ในแต่ละคอลัมน์
2. **ตรวจจับ data leakage** — คอลัมน์ `duration` (ความยาวการโทร) รู้ค่าได้ก็ต่อเมื่อโทรไปแล้ว จึงตัดออกจากฟีเจอร์หลัก (พิสูจน์ขนาดของ leakage ไว้ใน Appendix B)
3. **แบ่ง train/test แบบ time-based (ไม่สลับข้อมูล)** — เพราะข้อมูลเรียงตามเวลาจริง และ use case คือพยากรณ์ล่วงหน้า ไม่ใช่สุ่มแบ่ง (เทียบผลกับ random split ไว้ใน Appendix A)
4. **เปรียบเทียบ 6 โมเดล × 3 รูปแบบ cross-validation** (Dummy, LogisticRegression, RandomForest, HistGradientBoosting, XGBoost, LightGBM) ด้วย `StratifiedKFold(shuffle)`, `KFold(no shuffle)`, `TimeSeriesSplit`
5. **เลือกโมเดลจาก `TimeSeriesSplit`** (scheme ที่สมจริงที่สุด) แล้วประเมินบน test set ที่ไม่เคยเห็นเพียงครั้งเดียว
6. **ปรับ decision threshold** ตามเป้าหมายธุรกิจ โดยใช้ gains/lift curve แทนการตั้งค่าคงที่ เพราะสัดส่วนคลาสเปลี่ยนไปตามเวลา
7. **ตีความโมเดล** ด้วย permutation feature importance

## ผลลัพธ์หลัก

- โมเดลที่เลือก: **HistGradientBoosting** — Test ROC-AUC `0.680`, Test PR-AUC `0.468`
- **Data leakage ของ `duration`**: ถ้าใส่กลับเข้าไป ROC-AUC พุ่งจาก 0.680 → 0.787 (+0.107) — ยืนยันว่าห้ามใช้กับโมเดลที่ต้อง "เลือกเป้าหมายก่อนโทร"
- **Random split ให้ตัวเลขดูดีเกินจริง**: ROC-AUC จาก 0.680 (time-based) พุ่งเป็น 0.804 (random shuffle) — เป็น optimistic bias ที่ใช้งานจริงไม่ได้
- **Concept drift**: สัดส่วน deposit=yes เปลี่ยนจาก 6.7% (train) เป็น 31.6% (test) ทำให้ threshold ที่ tune จาก validation ใช้ตรง ๆ กับอนาคตไม่ได้ ต้องอิงกราฟ gains/lift ร่วมกับต้นทุน/มูลค่าทางธุรกิจแทน
- **ฟีเจอร์ที่สำคัญที่สุด**: `poutcome` (ผลแคมเปญครั้งก่อน) และ `month` (จังหวะเวลาติดต่อ)

รายละเอียดทั้งหมด (กราฟ, ตาราง, การตีความ) อยู่ในโน้ตบุ๊ก `BankMarketing.ipynb` และสรุปแบบนำเสนอใน `BankMarketing_Presentation.pptx`

## วิธีรัน

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
jupyter notebook BankMarketing.ipynb
```
