# fastapi==0.115.6
# uvicorn[standard]==0.32.1
# sqlalchemy==2.0.36
# pydantic==2.10.4
# pandas==2.2.3
# numpy==2.1.3
# statsmodels==0.14.4
# scikit-learn==1.5.2
# python-multipart==0.0.20

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./retail.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                       if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2
from sqlalchemy import Column, Integer, Float, String, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship
# Base is already defined above in this file


class Store(Base):
    __tablename__ = "stores"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    size = Column(String, default="small")  # small | medium | large
    sales = relationship("Sale", back_populates="store")


class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), index=True)
    date = Column(Date, index=True, nullable=False)
    product = Column(String, nullable=False)
    units = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    promotion = Column(Boolean, default=False)
    weather = Column(String, default="clear")
    store = relationship("Store", back_populates="sales")

# 3
from datetime import date
from typing import List
from pydantic import BaseModel, ConfigDict


class SaleCreate(BaseModel):
    store_id: int
    date: date
    product: str = "general"
    units: int
    unit_price: float
    promotion: bool = False
    weather: str = "clear"


class SaleOut(SaleCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)


class BulkSales(BaseModel):
    items: List[SaleCreate]


class TrainRequest(BaseModel):
    store_id: int


class ForecastPoint(BaseModel):
    date: date
    predicted_units: float


class ForecastResponse(BaseModel):
    store_id: int
    horizon_days: int
    points: List[ForecastPoint]
    model: str

# 4
from datetime import date
import numpy as np

HOLIDAYS = {(1, 1): "New Year", (12, 25): "Christmas",
            (12, 26): "Boxing Day", (11, 28): "Thanksgiving"}


def day_of_week_features(d: date):
    """Return [is_weekend, dow_sin, dow_cos] to capture weekly seasonality."""
    dow = d.weekday()
    return [1.0 if dow >= 5 else 0.0,
            float(np.sin(2 * np.pi * dow / 7)),
            float(np.cos(2 * np.pi * dow / 7))]


def is_holiday(d: date):
    return (d.month, d.day) in HOLIDAYS


def weather_factor(weather: str):
    return {"clear": 1.0, "rain": 0.85, "snow": 0.75, "hot": 1.15}.get(weather, 1.0)
#5

import logging
from datetime import timedelta
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from features import day_of_week_features, is_holiday, weather_factor
# Functions defined above in section #4

logger = logging.getLogger(__name__)


def _build_exog(dates, weather_series):
    rows = []
    for d, w in zip(dates, weather_series):
        rows.append(day_of_week_features(d) +
                    [1.0 if is_holiday(d) else 0.0, weather_factor(w)])
    return np.array(rows, dtype=float)


def train_and_forecast(sales, horizon=14, order=(1, 1, 1), seasonal=(1, 1, 1, 7)):
    """sales: list of (date, units, weather) sorted ascending."""
    if not sales or len(sales) < 7:
        raise ValueError("Need at least 7 days of history to forecast.")

    sales = sorted(sales, key=lambda r: r[0])
    dates = [r[0] for r in sales]
    units = np.array([r[1] for r in sales], dtype=float)
    weather = [r[2] for r in sales]

    exog = _build_exog(dates, weather)
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
    future_exog = _build_exog(future_dates, [weather[-1]] * horizon)

    try:
        model = SARIMAX(units, exog=exog, order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = np.maximum(res.get_forecast(steps=horizon, exog=future_exog).predicted_mean, 0.0)
        model_name = "SARIMAX"
    except Exception as exc:
        logger.warning("SARIMAX failed (%s); using naive seasonal baseline.", exc)
        base = units[-1]
        pred = np.array([max(0.0, base * (0.98 ** i)) for i in range(1, horizon + 1)])
        model_name = "naive-seasonal-baseline"

    return ([{"date": d.isoformat(), "predicted_units": float(p)}
             for d, p in zip(future_dates, pred)], model_name)
#6
from datetime import date
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

# Removed relative imports - all code is in this file

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Retail Sales Forecasting API",
              description="Forecast daily sales for local stores with SARIMAX.",
              version="1.0.0")

# Allow any website frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Retail Sales Forecast</title>
<style>
  body{font-family:system-ui,sans-serif;margin:2rem;max-width:800px}
  select,button{padding:.5rem;font-size:1rem}
  #out{margin-top:1rem;border-collapse:collapse;width:100%}
  #out td,#out th{border:1px solid #ccc;padding:.4rem;text-align:left}
</style>
</head>
<body>
<h1>Retail Sales Forecasting</h1>
<label>Store: <select id="store"></select></label>
<label>Days: <input id="days" type="number" value="14" min="1" max="90"></label>
<button onclick="load()">Forecast</button>
<table id="out"><thead><tr><th>Date</th><th>Predicted Units</th></tr></thead><tbody></tbody></table>
<script>
async function load(){
  const store = document.getElementById('store').value;
  const days = document.getElementById('days').value;
  const r = await fetch(`/forecast/${store}?horizon=${days}`);
  const data = await r.json();
  const tb = document.querySelector('#out tbody');
  tb.innerHTML = data.points.map(p =>
    `<tr><td>${p.date}</td><td>${p.predicted_units.toFixed(1)}</td></tr>`).join('');
}
(async () => {
  const stores = await (await fetch('/stores')).json();
  document.getElementById('store').innerHTML =
    stores.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sales", response_model=SaleOut, status_code=201)
def create_sale(payload: SaleCreate, db: Session = Depends(get_db)):
    if not db.get(Store, payload.store_id):
        raise HTTPException(404, f"Store {payload.store_id} not found")
    sale = Sale(**payload.model_dump())
    db.add(sale)
    db.commit()
    db.refresh(sale)
    return sale


@app.post("/sales/bulk", status_code=201)
def bulk_sales(payload: BulkSales, db: Session = Depends(get_db)):
    created = 0
    for item in payload.items:
        if not db.get(Store, item.store_id):
            raise HTTPException(404, f"Store {item.store_id} not found")
        db.add(Sale(**item.model_dump()))
        created += 1
    db.commit()
    return {"created": created}


@app.get("/sales", response_model=List[SaleOut])
def list_sales(store_id: Optional[int] = None, start: Optional[date] = None,
               end: Optional[date] = None, db: Session = Depends(get_db)):
    q = select(Sale)
    if store_id:
        q = q.where(Sale.store_id == store_id)
    if start:
        q = q.where(Sale.date >= start)
    if end:
        q = q.where(Sale.date <= end)
    return db.scalars(q.order_by(Sale.date)).all()


@app.get("/stores")
def list_stores(db: Session = Depends(get_db)):
    return [{"id": s.id, "name": s.name, "city": s.city, "size": s.size}
            for s in db.query(Store).all()]


@app.post("/forecast/train", response_model=ForecastResponse)
def train_store(payload: TrainRequest,
                horizon: int = Query(14, ge=1, le=90),
                db: Session = Depends(get_db)):
    if not db.get(Store, payload.store_id):
        raise HTTPException(404, f"Store {payload.store_id} not found")
    rows = (db.query(Sale)
              .filter(Sale.store_id == payload.store_id)
              .order_by(Sale.date).all())
    if len(rows) < 7:
        raise HTTPException(400, "Need at least 7 days of history for this store.")
    points, model_name = train_and_forecast([(r.date, r.units, r.weather) for r in rows],
                                            horizon=horizon)
    return ForecastResponse(store_id=payload.store_id, horizon_days=horizon,
                                    points=[ForecastPoint(**p) for p in points],
                                    model=model_name)


@app.get("/forecast/{store_id}", response_model=ForecastResponse)
def get_forecast(store_id: int, horizon: int = Query(14, ge=1, le=90),
                 db: Session = Depends(get_db)):
    return train_store(TrainRequest(store_id=store_id), horizon, db)
#7
if __name__ == "__main__":
    import random
    from datetime import date, timedelta
    # Removed relative imports - all code is in this file

    random.seed(42)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    if db.query(Store).count() == 0:
        db.add_all([Store(name="Corner Grocery", city="Portland", size="small"),
                     Store(name="Main St Market", city="Portland", size="medium"),
                     Store(name="Harbor Convenience", city="Seattle", size="small")])
        db.commit()

    weather_pool = ["clear", "rain", "snow", "hot"]
    start = date(2025, 1, 1)
    for store in db.query(Store).all():
        base = {"small": 30, "medium": 60, "large": 100}[store.size]
        for i in range(365):
            d = start + timedelta(days=i)
            weekend = 1.5 if d.weekday() >= 5 else 1.0
            holiday = 1.8 if is_holiday(d) else 1.0
            weather = random.choice(weather_pool)
            promo = random.random() < 0.15
            promo_mult = 1.6 if promo else 1.0
            noise = random.uniform(0.7, 1.3)
            units = int(base * weekend * holiday * weather_factor(weather) * promo_mult * noise)
            db.add(Sale(store_id=store.id, date=d, product="general",
                        units=max(1, units),
                        unit_price=round(random.uniform(3, 15), 2),
                        promotion=promo, weather=weather))
    db.commit()
    print("Seeded 3 stores with 365 days of sales each.")

#9
# pip install -r requirements.txt
# python app/seed.py
# uvicorn app.main:app --reload