# EPL Match Predictor

Dashboard du doan ket qua cac tran dau English Premier League bang du lieu truoc tran. Project nay tap trung vao bai toan phan loai 3 ket qua:

- `Home Win`: doi chu nha thang
- `Draw`: hoa
- `Away Win`: doi khach thang

Model khong su dung du lieu trong tran hoac sau tran de tao feature cho tran can du doan. Cac feature duoc tinh tu lich su truoc ngay thi dau: phong do, ELO, ban thang/thua, shots on target, thanh tich san nha/san khach, head-to-head, bang xep hang, dong luc cuoi mua va ti le hoa.

## Trang thai hien tai

Model dang duoc luu trong `models/model_best.pkl`:

| Thong tin | Gia tri |
|---|---:|
| Model | Robust Logistic Regression |
| Version | v18 |
| Validation accuracy | 55.88% |
| Test season accuracy | 50.53% |
| GW31-38 weighted accuracy | 37/71 = 52.11% |
| Draw boost | 0.10 |
| So feature | 20 |

Ket qua backtest theo gameweek hien co trong `predictions/accuracy_comparison.csv`:

| Gameweek | Accuracy |
|---:|---:|
| GW31 | 100% |
| GW32 | 60% |
| GW33 | 40% |
| GW34 | 60% |
| GW35 | 60% |
| GW36 | 30% |
| GW37 | 60% |
| GW38 | 50% |

## Chuc nang chinh

- Thu thap du lieu EPL tu `football-data.co.uk`.
- Lam sach du lieu va tao feature truoc tran.
- Train nhieu model va chon model tot nhat theo walk-forward validation.
- Du doan ket qua tung gameweek.
- Backtest GW31-GW38 cua mua 2025/26.
- Dashboard React hien thi:
  - du doan tung tran;
  - xac suat Home/Draw/Away;
  - nhom tran ro xu huong, can bang, cua hoa dang chu y;
  - thong tin chi tiet truoc tran khi click vao tung tran;
  - accuracy theo gameweek;
  - confusion matrix va feature importance;
  - bang xep hang.

## Cau truc project

```text
.
├── collect_data.py              # Download/cap nhat du lieu EPL
├── clean_data.py                # Tao data/epl_clean.csv va feature truoc tran
├── train_model.py               # Train, so sanh model, luu model tot nhat
├── predict.py                   # Du doan fixtures/gameweek
├── evaluate_all_gws.py          # Backtest GW31-GW38
├── generate_reports.py          # Tao lai reports tu model hien tai
├── retrain.py                   # Workflow retrain sau moi gameweek
├── fetch_fixtures.py            # Lay lich thi dau tu football-data.org
├── backend/
│   └── main.py                  # FastAPI backend
├── frontend/
│   ├── src/                     # React dashboard
│   └── package.json
├── data/
│   ├── epl_raw.csv              # Du lieu goc da gop
│   └── epl_clean.csv            # Du lieu sau khi tao feature
├── models/
│   ├── model_best.pkl           # Model dang dung
│   ├── scaler.pkl
│   └── model_compare.csv
├── predictions/
│   ├── gw*_fixtures.csv
│   ├── gw*_predictions.csv
│   └── accuracy_comparison.csv
└── reports/
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Cai dat

Yeu cau:

- Python 3.10+
- Node.js 18+
- npm

Tao moi truong Python:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy requests scikit-learn xgboost lightgbm matplotlib seaborn fastapi uvicorn
```

Cai frontend:

```bash
cd frontend
npm install
cd ..
```

## Chay nhanh dashboard

Repo da co san `data/`, `models/`, `predictions/` va `reports/`, nen co the chay dashboard ngay ma khong can train lai.

Terminal 1 - backend:

```bash
source venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000
```

Terminal 2 - frontend:

```bash
cd frontend
npm run dev
```

Mo trinh duyet:

```text
http://localhost:5173
```

Frontend mac dinh goi API tai:

```text
http://localhost:8000/api
```

## Chay lai toan bo pipeline

### 1. Thu thap du lieu

Download lai tat ca cac mua:

```bash
source venv/bin/activate
python collect_data.py
```

Neu chi muon cap nhat mua hien tai, vi du sau khi vua da xong mot vong:

```bash
python collect_data.py --current-only
```

Lenh nay chi download lai mua moi nhat va replace phan mua do trong `data/epl_raw.csv`, tranh append trung tran.

### 2. Lam sach va tao feature

```bash
python clean_data.py
```

Output:

```text
data/epl_clean.csv
data/epl_seasons_clean/
```

### 3. Train model

```bash
python train_model.py
```

Output:

```text
models/model_best.pkl
models/scaler.pkl
models/model_compare.csv
reports/confusion_matrix.png
reports/feature_importance.png
```

Neu moi truong terminal bi loi khi ve bieu do, co the dung backend matplotlib khong GUI:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp python train_model.py
```

### 4. Tao lai report theo model hien tai

Dung khi muon dam bao anh trong `reports/` khop voi `models/model_best.pkl`:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp python generate_reports.py
```

### 5. Du doan gameweek

Neu da co fixture file trong `predictions/gw{N}_fixtures.csv`, chay:

```bash
python predict.py
```

Output:

```text
predictions/gw{N}_predictions.csv
```

### 6. Backtest GW31-GW38

```bash
python evaluate_all_gws.py
```

Output:

```text
predictions/accuracy_comparison.csv
```

Dashboard se uu tien doc file nay de hien thi accuracy moi.

## Workflow sau moi gameweek

Sau khi mot vong dau ket thuc:

```bash
source venv/bin/activate
python collect_data.py --current-only
python clean_data.py
python retrain.py
python predict.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp python generate_reports.py
```

Neu can lay lich thi dau sap toi tu football-data.org:

```bash
python fetch_fixtures.py
```

Luu y: `fetch_fixtures.py` can API key cua football-data.org.

## API backend

Backend FastAPI cung cap cac endpoint:

| Endpoint | Mo ta |
|---|---|
| `GET /api/predictions/latest` | Du doan gameweek moi nhat co file prediction |
| `GET /api/predictions/{gameweek}` | Du doan cua mot gameweek cu the |
| `GET /api/predictions/gameweeks` | Danh sach gameweek co prediction |
| `GET /api/accuracy` | Accuracy theo gameweek |
| `GET /api/standings` | Bang xep hang local/API |
| `GET /api/model/info` | Thong tin model hien tai |
| `GET /reports/confusion_matrix.png` | Anh confusion matrix |
| `GET /reports/feature_importance.png` | Anh feature importance |

## Feature engineering

Mot so nhom feature quan trong:

- Form gan day:
  - `wform_diff`
  - `adj_form_diff`
  - `momentum_diff`
- Tan cong/phong ngu:
  - `scored_diff`
  - `conceded_diff`
  - `sot_diff`
  - `cs_diff`
- Suc manh doi bong:
  - `elo_diff`
  - `season_gd_diff`
  - `ppg_diff`
- San nha/san khach:
  - `venue_form_diff`
- Lich su doi dau:
  - `h2h_dominance`
- Cuoi mua va dong luc:
  - `season_progress`
  - `gap_top_diff`
  - `gap_rel_diff`
  - `motivation_diff`
  - `low_motivation`
  - `draw_rate_avg`

Tat ca feature duoc tinh tu cac tran da dien ra truoc ngay thi dau cua tran can du doan.

## Model

`train_model.py` so sanh cac model:

- Logistic Regression
- Robust Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- LightGBM
- Voting Ensemble

Model tot nhat hien tai la `Robust Logistic Regression`, duoc luu trong:

```text
models/model_best.pkl
```

Model co them `draw_boost` de tang kha nang du doan hoa, vi ket qua hoa la lop kho nhat trong bai toan 3 lop EPL.

## Frontend

Frontend duoc viet bang React + Vite.

Tabs chinh:

- `Du doan`: xem du doan tung gameweek, nhom tran theo muc do ro xu huong/can bang/cua hoa, xem chi tiet truoc tran.
- `Do chinh xac`: accuracy theo gameweek.
- `Phan tich`: model info, confusion matrix, feature importance.
- `Bang xep hang`: bang xep hang EPL tinh tu local data hoac API.

Build frontend:

```bash
cd frontend
npm run build
```

## Luu y ve du lieu va ket qua

- Du lieu chinh lay tu `football-data.co.uk`.
- Ket qua du doan bong da co nhieu bat dinh; accuracy khong nen hieu la dam bao ket qua.
- Draw la lop kho nhat vi xac suat hoa thuong nam giua va it khi la xac suat cao nhat.
- Dashboard hien thi xac suat va cac thong tin truoc tran de nguoi dung hieu ly do model dua ra du doan, khong phai de khang dinh chac chan ti so.

