"""Data audit, cleaning, validation, and exploratory analysis for Instamart."""

from pathlib import Path
from itertools import combinations
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DB_FILE = ROOT / "data" / "instamart.db"


def audit():
    """Audit the raw customer, product, order, and ticket datasets."""
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)

    tables = {"customers": customers, "products": products, "orders": orders}

    for name, df in tables.items():
        print(f"\n{'=' * 50}")
        print(f"TABLE: {name.upper()}")
        print("=" * 50)
        print("\nShape:", df.shape)
        print("\nData types:")
        print(df.dtypes)
        missing = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2),
        })
        print("\nMissing values:")
        print(missing.to_string())

    print("\n========== ORDERS AUDIT ==========")
    numeric_amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    print("Raw order rows:", len(orders))
    print("Exact duplicate rows:", orders.duplicated().sum())
    print("Pending rows:", orders["status"].eq("pending").sum())
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("Non-numeric amounts:", numeric_amount.isna().sum())
    print("Negative delivery times:", delivery.lt(0).sum())
    blank_delivery = (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype(str).str.strip().eq("")
    )
    print("Blank delivery times:", blank_delivery.sum())

    print("\n========== CUSTOMERS AUDIT ==========")
    age = pd.to_numeric(customers["age"], errors="coerce")
    print(
        "Blank ages:",
        customers["age"].isna().sum()
        + customers["age"].astype(str).str.strip().eq("").sum(),
    )
    print("Ages outside 10-100:", ((age < 10) | (age > 100)).sum())
    print("Distinct raw city spellings:", customers["city"].nunique())
    gender = customers["gender"].astype(str).str.strip()
    print(
        "Distinct non-blank raw gender spellings:",
        gender[gender.ne("") & gender.ne("nan")].nunique(),
    )
    print(
        "Blank genders:",
        customers["gender"].isna().sum() + gender.eq("").sum(),
    )

    print("\n========== PRODUCTS AUDIT ==========")
    print("Product rows:", len(products))
    print("Distinct product IDs:", products["product_id"].nunique())
    print("\nTickets loaded with keep_default_na=False:", len(tickets))


def clean_data():
    """Clean customers, orders, and products using the original rules."""

    def clean_customers():
        raw_file = RAW_DIR / "customers.csv"
        output_file = CLEAN_DIR / "customers.csv"
        customers = pd.read_csv(raw_file)
        print("Original shape:", customers.shape)

        customers["city"] = (
            customers["city"].astype("string").str.strip().str.title()
        )
        gender = customers["gender"].astype("string").str.strip().str.lower()
        male_values = {"m", "male", "man", "men"}
        female_values = {"f", "female", "woman", "women"}
        customers["gender"] = gender.map(
            lambda value: (
                "male" if value in male_values
                else "female" if value in female_values
                else "unknown"
            )
        )
        customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
        customers.loc[~customers["age"].between(10, 100), "age"] = float("nan")
        customers["age"] = customers["age"].fillna(
            customers.groupby("city")["age"].transform("median")
        )
        customers["age"] = customers["age"].astype(int)
        customers["signup_date"] = pd.to_datetime(
            customers["signup_date"], errors="coerce"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        customers.to_csv(output_file, index=False)

        print("\n========== CLEANED CUSTOMERS ==========")
        print("\nCleaned shape:", customers.shape)
        print("\nCity counts:")
        print(customers["city"].value_counts().sort_index())
        print("\nGender counts:")
        print(customers["gender"].value_counts())
        print("\nMissing ages:", customers["age"].isna().sum())
        print("Minimum age:", customers["age"].min())
        print("Maximum age:", customers["age"].max())
        print("Missing signup dates:", customers["signup_date"].isna().sum())
        

    def clean_orders():
        orders_file = RAW_DIR / "orders.csv"
        products_file = RAW_DIR / "products.csv"
        output_file = CLEAN_DIR / "orders.csv"
        orders = pd.read_csv(orders_file)
        products = pd.read_csv(products_file)
        print("\n========== CLEANED ORDERS ==========")

        orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
        before_duplicates = len(orders)
        orders = orders.drop_duplicates()
        print("Exact duplicate rows removed:", before_duplicates - len(orders))
        orders = orders.sort_values("order_ts")
        orders = orders.drop_duplicates(subset="order_id", keep="last")
        print("Rows after keeping latest order:", len(orders))
        orders["order_amount"] = pd.to_numeric(
            orders["order_amount"], errors="coerce"
        )
        orders = orders.merge(
            products[["product_id", "price_inr", "category"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        calculated_amount = (
            orders["price_inr"]
            * orders["quantity"]
            * (1 - orders["discount_pct"] / 100)
        ).round(2)
        orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)
        orders["delivery_minutes"] = pd.to_numeric(
            orders["delivery_minutes"], errors="coerce"
        )
        orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")
        completed_mask = orders["status"].eq("completed")
        category_median = (
            orders.loc[completed_mask]
            .groupby("category")["delivery_minutes"]
            .transform("median")
        )
        orders.loc[completed_mask, "delivery_minutes"] = (
            orders.loc[completed_mask, "delivery_minutes"].fillna(category_median)
        )
        orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        orders.to_csv(output_file, index=False)

        print("\nCleaned orders shape:", orders.shape)
        print("\nStatus counts:")
        print(orders["status"].value_counts())
        if orders["status"].eq("pending").sum() == 0:
            print("\nNo Pending rows")
        print("\nMissing order amounts:", orders["order_amount"].isna().sum())
        print(
            "Missing delivery times in completed orders:",
            orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum(),
        )
        print(
            "Missing delivery times in cancelled orders:",
            orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum(),
        )
        completed_total = orders.loc[
            orders["status"].eq("completed"), "order_amount"
        ].sum()
        print(f"Completed order amount total: INR {completed_total:.2f}")
        

    def clean_products():
        raw_file = RAW_DIR / "products.csv"
        output_file = CLEAN_DIR / "products.csv"
        products = pd.read_csv(raw_file)
        print("Original products shape:", products.shape)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        products.to_csv(output_file, index=False)
        print("\nCleaned products shape:", products.shape)
        print("Duplicate product IDs:", products["product_id"].duplicated().sum())
        

    clean_customers()
    clean_orders()
    clean_products()




def store_and_validate():
    """Save the cleaned tables to SQLite and assert the reference checkpoints."""
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        orders.to_sql("orders", con, if_exists="replace", index=False)
        tables_df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
        )
        print("Database tables:")
        print(tables_df)

    assert len(customers) == 1200
    assert customers["city"].nunique() == 6
    assert customers["age"].isna().sum() == 0
    assert customers["age"].between(18, 60).all()
    assert len(products) == 40
    assert products["product_id"].nunique() == 40
    assert len(orders) == 12309
    assert orders["order_id"].nunique() == 12309
    status_counts = orders["status"].value_counts()
    assert status_counts.get("completed", 0) == 11110
    assert status_counts.get("cancelled", 0) == 1199
    assert status_counts.get("pending", 0) == 0

    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    assert orders["order_amount"].isna().sum() == 0
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0
    assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199
    total_revenue = orders.loc[completed, "order_amount"].sum()
    assert abs(total_revenue - 2998668.80) < 0.01

    timestamps = pd.to_datetime(
        orders["order_ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    assert timestamps.notna().all()

    with sqlite3.connect(DB_FILE) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"customers", "products", "orders"}.issubset(tables)
        assert pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM customers", con
        ).iloc[0]["n"] == len(customers)
        assert pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM products", con
        ).iloc[0]["n"] == len(products)
        assert pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders", con
        ).iloc[0]["n"] == len(orders)
        non_text_timestamps = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM orders WHERE typeof(order_ts) != 'text'", con
        ).iloc[0]["n"]
        assert non_text_timestamps == 0

    print("\nALL CHECKPOINTS PASSED")


def sql_analysis():
    """Run the seven SQL assignment queries and display their reference results."""

    def connect_database():
        return sqlite3.connect(DB_FILE)

    def execute_queries(conn):
        query_file = ROOT / "sql" / "queries.sql"
        query_text = query_file.read_text(encoding="utf-8")
        queries = [
            query.strip()
            for query in query_text.split(";")
            if query.strip()
        ]
        assert len(queries) == 7, "sql/queries.sql must contain exactly seven queries."

        results = {}
        for index, query in enumerate(queries, start=1):
            results[f"Q{index}"] = pd.read_sql_query(query, conn)

        return results

    def display_results(results):
        print("\n========== SQL ANALYSIS ==========")

        q1 = results["Q1"]
        print("\n----- Query 1 Result -----")
        print("Customer Count:", len(q1))

        q2 = results["Q2"]
        print("\n----- Query 2 Result -----")
        print(q2.assign(
            revenue=q2["revenue"].map(lambda value: f"{value:,.2f}")
        ).to_string(index=False))
        print(f"Row count: {len(q2)}")
        print(f"Total revenue: {q2['revenue'].sum():,.2f}")
        largest_cell = q2.loc[q2["revenue"].idxmax()]
        print(
            "Largest cell: "
            f"{largest_cell['city']} / {largest_cell['category']} - "
            f"INR {largest_cell['revenue']:,.2f}"
        )

        q3 = results["Q3"]
        print("\n----- Query 3 Result -----")
        print("Product Count:", len(q3))

        q4 = results["Q4"]
        print("\n----- Query 4 Result -----")
        print("Customer Count:", len(q4))

        q5 = results["Q5"]
        print("\n----- Query 5 Result -----")
        print("Customer Count:", len(q5))
        if not q5.empty:
            print(
                "Average customer spend: "
                f"INR {q5['average_spending'].iloc[0]:,.2f}"
            )

        q6 = results["Q6"]
        print("\n----- Query 6 Result -----")
        print("Top Customers:")
        print(
            q6[q6["rank_position"] == 1][
                ["city", "customer_id", "total_spend"]
            ].to_string(index=False)
        )
        print("RANK Differences:",
                (q6["rank_position"] != q6["row_number_position"]).sum())
        

        q7 = results["Q7"]
        print("\nQ7 - Daily completed revenue and running total:")
        print("Daily row count:", len(q7))
        if not q7.empty:
            last_row = q7.iloc[-1]
            print(
                f"Final cumulative revenue: "
                f"INR {last_row['cumulative_revenue']:,.2f}"
            )

    conn = connect_database()
    results = execute_queries(conn)
    display_results(results)
    conn.close()


def validate():
    """Run the Pandas cross-check and exploratory analysis on completed orders."""
    with sqlite3.connect(DB_FILE) as con:
        analysis_orders = pd.read_sql_query(
            """
            SELECT o.order_id, o.customer_id, o.product_id, c.city, p.category,
                   o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN products p ON o.product_id = p.product_id
            WHERE o.status = 'completed'
            """,
            con,
        )
        sql_total = pd.read_sql_query(
            "SELECT SUM(order_amount) AS total_revenue FROM orders WHERE status = 'completed'",
            con,
        ).iloc[0]["total_revenue"]

    print("\n========== PANDAS CROSS-CHECK ==========")
    pivot = pd.pivot_table(
        analysis_orders,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        margins=True,
        margins_name="Grand Total",
        fill_value=0,
    )
    print("\nRevenue by city and category:")
    print(pivot)
    pandas_total = pivot.loc["Grand Total", "Grand Total"]
    print(f"\nSQL total: INR {sql_total:,.2f}")
    print(f"Pandas grand total: INR {pandas_total:,.2f}")
    assert round(sql_total, 2) == round(pandas_total, 2)
    print("Total check: PASS")

    city_summary = analysis_orders.groupby("city").agg(
        total_revenue=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count")).round(2)
    
    print("\nCity-wise completed-order summary:")
    print(city_summary)

    # Reference checks for the two cities specified in the assignment.
    assert abs(city_summary.loc["Bengaluru", "total_revenue"] - 523643.40) < 0.01
    assert abs(city_summary.loc["Bengaluru", "average_order_amount"] - 276.47) < 0.01
    assert city_summary.loc["Bengaluru", "order_count"] == 1894
    assert abs(city_summary.loc["Mumbai", "total_revenue"] - 479058.50) < 0.01
    assert abs(city_summary.loc["Mumbai", "average_order_amount"] - 297.00) < 0.01
    assert city_summary.loc["Mumbai", "order_count"] == 1613

    print("\n========== EXPLORATORY DATA ANALYSIS ==========")
    amount = analysis_orders["order_amount"]
    print("\nDescriptive statistics:")
    skewness = amount.skew()
    print(f"Mean: {amount.mean():.2f}")
    print(f"Median: {amount.median():.2f}")
    print(f"Skewness: {skewness:.2f}")
    if skewness > 0:
        print("Distribution: Right-skewed")
    elif skewness < 0:
        print("Distribution: Left-skewed")
    else:
        print("Distribution: Symmetrical")

    q1 = amount.quantile(0.25)
    q2 = amount.quantile(0.50)
    q3 = amount.quantile(0.75)
    print("\nQuartiles:")
    print(f"Q1: {q1:.2f}")
    print(f"Q2 (Median): {q2:.2f}")
    print(f"Q3: {q3:.2f}")
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    print(f"\nIQR: {iqr:.2f}")
    print(f"Lower fence: {lower_fence:.2f}")
    print(f"Upper fence: {upper_fence:.2f}")
    below_outliers = (amount < lower_fence).sum()
    above_outliers = (amount > upper_fence).sum()
    print("\nOutlier counts:")
    print("Below lower fence:", below_outliers)
    print("Above upper fence:", above_outliers)

    print("\n========== CORRELATION MATRIX ==========")
    columns = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    correlation = analysis_orders[columns].corr()
    print(correlation.round(3))
    pairs = [
        (a, b, abs(correlation.loc[a, b]), correlation.loc[a, b])
        for a, b in combinations(columns, 2)
    ]
    largest_pair = max(pairs, key=lambda item: item[2])
    print("\nLargest correlation pair by absolute value:")
    print(largest_pair[0], "and", largest_pair[1])
    print(f"Correlation: {largest_pair[3]:+.3f}")
    amount_quantity_corr = correlation.loc["order_amount", "quantity"]
    print(
        "Correlation between order_amount and quantity: "
        f"{amount_quantity_corr:+.3f}"
    )
    


if __name__ == "__main__":
    audit()
    clean_data()
    store_and_validate()
    sql_analysis()
    validate()
