
import pandas as pd
import numpy as np
from pathlib import Path


class DataAnalysisService:

    def _find_column(self, df, aliases):
        """Find a column using normalized names and aliases."""
        normalized = {
            str(column).lower().strip().replace(" ", "_"): column
            for column in df.columns
        }

        for alias in aliases:
            if alias in normalized:
                return normalized[alias]

        return None

    def _safe_value(self, value):
        """Convert Pandas and NumPy values into JSON-safe values."""
        if value is None:
            return None

        if isinstance(value, dict):
            return {
                str(k): self._safe_value(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [self._safe_value(v) for v in value]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None

        if isinstance(value, (np.bool_,)):
            return bool(value)

        if pd.isna(value):
            return None

        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()

        return value

    def _group_analysis(self, df, category_column, revenue_column,
                        profit_column=None, limit=10):
        """Calculate revenue and profit by a category."""
        if category_column is None or revenue_column is None:
            return []

        working = df.copy()

        working[revenue_column] = pd.to_numeric(
            working[revenue_column], errors="coerce"
        )

        aggregations = {
            "revenue": (revenue_column, "sum")
        }

        if profit_column is not None:
            working[profit_column] = pd.to_numeric(
                working[profit_column], errors="coerce"
            )
            aggregations["profit"] = (profit_column, "sum")

        grouped = (
            working.groupby(category_column, dropna=False)
            .agg(**aggregations)
            .reset_index()
        )

        grouped = grouped.sort_values(
            "revenue", ascending=False
        ).head(limit)

        return [
            {
                "category": str(row[category_column]),
                "revenue": float(row["revenue"]),
                **(
                    {"profit": float(row["profit"])}
                    if "profit" in grouped.columns else {}
                )
            }
            for _, row in grouped.iterrows()
        ]

    def analyze_csv(self, file_path: str) -> dict:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError("The uploaded CSV file was not found.")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        if len(df.columns) == 0:
            raise ValueError("No columns found in the CSV file.")

        # -------------------------------------------------
        # 1. BASIC DATASET INFORMATION
        # -------------------------------------------------

        rows, columns = df.shape

        missing_values = {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
        }

        duplicate_rows = int(df.duplicated().sum())

        numeric_df = df.select_dtypes(include=np.number)

        numeric_summary = {}

        for column in numeric_df.columns:
            series = numeric_df[column].dropna()

            numeric_summary[str(column)] = {
                "total": float(series.sum()) if not series.empty else 0,
                "average": float(series.mean()) if not series.empty else None,
                "minimum": float(series.min()) if not series.empty else None,
                "maximum": float(series.max()) if not series.empty else None
            }

        # -------------------------------------------------
        # 2. IDENTIFY BUSINESS COLUMNS
        # -------------------------------------------------

        revenue_column = self._find_column(
            df, ["revenue", "sales", "total_revenue", "sales_amount"]
        )

        cost_column = self._find_column(
            df, ["cost", "total_cost", "expense", "expenses"]
        )

        profit_column = self._find_column(
            df, ["profit", "net_profit", "gross_profit"]
        )

        region_column = self._find_column(
            df, ["region", "area", "territory", "location"]
        )

        product_column = self._find_column(
            df, ["product", "product_name", "item", "category"]
        )

        date_column = self._find_column(
            df, ["date", "order_date", "sales_date", "transaction_date"]
        )

        business_kpis = {}

        # -------------------------------------------------
        # 3. FINANCIAL KPI CALCULATIONS
        # -------------------------------------------------

        revenue = None
        cost = None
        profit = None

        if revenue_column:
            revenue = pd.to_numeric(
                df[revenue_column], errors="coerce"
            )

            business_kpis["total_revenue"] = float(revenue.sum())

        if cost_column:
            cost = pd.to_numeric(
                df[cost_column], errors="coerce"
            )

            business_kpis["total_cost"] = float(cost.sum())

        if profit_column:
            profit = pd.to_numeric(
                df[profit_column], errors="coerce"
            )

            business_kpis["total_profit"] = float(profit.sum())

        elif revenue_column and cost_column:
            profit = revenue - cost

            business_kpis["total_profit"] = float(profit.sum())
            business_kpis["calculated_profit"] = float(profit.sum())

        if revenue is not None and profit is not None:
            total_revenue = revenue.sum()
            total_profit = profit.sum()

            if total_revenue != 0:
                business_kpis["profit_margin"] = round(
                    float((total_profit / total_revenue) * 100), 2
                )
            else:
                business_kpis["profit_margin"] = None

        if revenue is not None:
            business_kpis["average_revenue"] = (
                float(revenue.mean())
                if revenue.notna().any() else None
            )

        # -------------------------------------------------
        # 4. REGIONAL PERFORMANCE
        # -------------------------------------------------

        regional_performance = self._group_analysis(
            df,
            region_column,
            revenue_column,
            profit_column if profit is not None else None
        )

        # -------------------------------------------------
        # 5. PRODUCT PERFORMANCE
        # -------------------------------------------------

        product_performance = self._group_analysis(
            df,
            product_column,
            revenue_column,
            profit_column if profit is not None else None
        )

        # -------------------------------------------------
        # 6. MONTHLY REVENUE TRENDS
        # -------------------------------------------------

        monthly_trends = []

        if date_column and revenue_column:
            dates = pd.to_datetime(
                df[date_column], errors="coerce"
            )

            trend_df = pd.DataFrame({
                "date": dates,
                "revenue": revenue
            }).dropna()

            if not trend_df.empty:
                trend_df["month"] = (
                    trend_df["date"].dt.to_period("M").astype(str)
                )

                monthly = (
                    trend_df.groupby("month")["revenue"]
                    .sum()
                    .reset_index()
                )

                monthly_trends = [
                    {
                        "month": str(row["month"]),
                        "revenue": float(row["revenue"])
                    }
                    for _, row in monthly.iterrows()
                ]

                if len(monthly) >= 2:
                    previous = monthly.iloc[-2]["revenue"]
                    current = monthly.iloc[-1]["revenue"]

                    if previous != 0:
                        business_kpis["monthly_revenue_change_percent"] = round(
                            float((current - previous) / abs(previous) * 100),
                            2
                        )

        # -------------------------------------------------
        # 7. AUTOMATED BUSINESS INSIGHTS
        # -------------------------------------------------

        insights = []

        if revenue is not None and profit is not None:
            margin = business_kpis.get("profit_margin")

            if margin is not None:
                insights.append(
                    f"Overall profit margin is {margin}%."
                )

                if margin < 10:
                    insights.append(
                        "Profit margin is below 10%. Review costs and pricing."
                    )
                elif margin >= 30:
                    insights.append(
                        "The business has a profit margin of at least 30%."
                    )

        if regional_performance:
            best_region = regional_performance[0]

            insights.append(
                f"Highest revenue-generating region: "
                f"{best_region['category']} "
                f"with revenue of {best_region['revenue']:.2f}."
            )

            lowest_region = min(
                regional_performance,
                key=lambda item: item["revenue"]
            )

            if len(regional_performance) > 1:
                insights.append(
                    f"Lowest revenue among the analyzed regions: "
                    f"{lowest_region['category']} "
                    f"with revenue of {lowest_region['revenue']:.2f}."
                )

        if product_performance:
            best_product = max(
                product_performance,
                key=lambda item: (
                    item.get("profit", item["revenue"])
                )
            )

            insights.append(
                f"Top product by "
                f"{'profit' if 'profit' in best_product else 'revenue'}: "
                f"{best_product['category']}."
            )

        if "monthly_revenue_change_percent" in business_kpis:
            change = business_kpis["monthly_revenue_change_percent"]

            if change < 0:
                insights.append(
                    f"Revenue declined by {abs(change)}% compared "
                    "with the previous month in the dataset."
                )
            elif change > 0:
                insights.append(
                    f"Revenue increased by {change}% compared "
                    "with the previous month in the dataset."
                )

        # -------------------------------------------------
        # 8. DATA QUALITY WARNINGS
        # -------------------------------------------------

        data_quality_warnings = []

        if duplicate_rows > 0:
            data_quality_warnings.append(
                f"Dataset contains {duplicate_rows} duplicate rows."
            )

        total_missing = sum(missing_values.values())

        if total_missing > 0:
            data_quality_warnings.append(
                f"Dataset contains {total_missing} missing values."
            )

        if revenue_column is None:
            data_quality_warnings.append(
                "Revenue column not detected. Financial revenue KPIs "
                "could not be calculated."
            )

        # -------------------------------------------------
        # 9. FINAL RESPONSE
        # -------------------------------------------------

        result = {
            "dataset_name": path.name,
            "rows": rows,
            "columns": columns,
            "column_names": df.columns.tolist(),
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "numeric_summary": numeric_summary,
            "business_kpis": business_kpis,
            "regional_performance": regional_performance,
            "product_performance": product_performance,
            "monthly_trends": monthly_trends,
            "business_insights": insights,
            "data_quality_warnings": data_quality_warnings,
            "preview": df.head(5).to_dict(orient="records")
        }

        return self._safe_value(result)
