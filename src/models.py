
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF_DATE = pd.Timestamp("2025-09-01")
LABEL_END_DATE = pd.Timestamp("2025-10-01")

_MODEL_CONTEXT = None


def rfm_and_segments():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    completed = orders[
        orders["status"] == "completed"
    ].copy()

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()

    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(
    OUTPUT_DIR / "customer_rfm.csv",
    index=False
)

    print("\nRFM table shape:", rfm.shape)
    print("\nRecency statistics:")
    print(rfm["recency"].agg(["median", "min", "max"]))
    print("\nFrequency statistics:")
    print(rfm["frequency"].agg(["mean", "max"]).round(2))
    print("\nMonetary statistics:")
    print(rfm["monetary"].agg(["mean", "max"]).map(lambda x: f"INR {x:,.2f}"))

    features = rfm[
        ["recency", "frequency", "monetary"]
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    results = []

    for k in range(2, 9):
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )

        cluster_labels = kmeans.fit_predict(scaled_features)

        results.append({
            "K": k,
            "Inertia": round(kmeans.inertia_, 1),
            "Silhouette Score": round(
                silhouette_score(scaled_features, cluster_labels),
                3,
            ),
        })

    results_df = pd.DataFrame(results)
    print("\nK-means Evaluation Results:")
    print(results_df.to_string(index=False))

    k = 4

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    rfm["segment_id"] = kmeans.fit_predict(scaled_features)

    print("\nSelected K:", k)

    segment_profile = rfm.groupby("segment_id").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).round(2)

    print("\nCustomer Segment Profiles:")
    print(segment_profile.to_string(formatters={"avg_monetary": lambda x: f"INR {x:,.2f}"}))

    remaining_segments = list(segment_profile.index)

    premium_segment = segment_profile[
        "avg_monetary"
    ].idxmax()

    segment_name_mapping = {
        premium_segment: "Premium High-Spenders"
    }

    remaining_segments.remove(premium_segment)

    at_risk_segment = segment_profile.loc[
        remaining_segments, "avg_recency"
    ].idxmax()

    segment_name_mapping[
        at_risk_segment
    ] = "At-Risk Inactive Customers"

    remaining_segments.remove(at_risk_segment)

    frequent_segment = segment_profile.loc[
        remaining_segments, "avg_frequency"
    ].idxmax()

    segment_name_mapping[
        frequent_segment
    ] = "Frequent Engaged Customers"

    remaining_segments.remove(frequent_segment)

    occasional_segment = remaining_segments[0]

    segment_name_mapping[
        occasional_segment
    ] = "Occasional Value Seekers"

    segment_actions = {
        "Occasional Value Seekers":
            "Encourage repeat purchases with personalized offers",
        "Premium High-Spenders":
            "Provide loyalty rewards and exclusive offers",
        "Frequent Engaged Customers":
            "Maintain engagement with loyalty and repeat-order benefits",
        "At-Risk Inactive Customers":
            "Target with win-back campaigns and reactivation offers",
    }

    rfm["segment_name"] = rfm["segment_id"].map(
        segment_name_mapping
    )

    rfm["is_premium"] = (
        rfm["segment_id"] == premium_segment
    )

    print("\nSegment Names and Recommended Actions:")

    for segment_id in segment_profile.index:
        name = segment_name_mapping[segment_id]
        print(
            f"Segment {segment_id}: {name} - "
            f"{segment_actions[name]}"
        )

    print("\nPremium Segment:", premium_segment)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    print("\nPCA Explained Variance:")
    print(
        "Component 1:",
        round(pca.explained_variance_ratio_[0], 3),
    )
    print(
        "Component 2:",
        round(pca.explained_variance_ratio_[1], 3),
    )
    print(
        "Total explained variance:",
        round(pca.explained_variance_ratio_.sum(), 3),
    )

    rfm["pca_1"] = pca_result[:, 0]
    rfm["pca_2"] = pca_result[:, 1]

    output = rfm[[
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]]

    output.to_csv(
        OUTPUT_DIR / "customer_segments.csv",
        index=False,
    )


def build_lapse_dataset():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    before_cutoff = orders[
        orders["order_ts"] < CUTOFF_DATE
    ].copy()

    completed_before = before_cutoff[
        before_cutoff["status"] == "completed"
    ].copy()

    cohort_ids = completed_before[
        "customer_id"
    ].unique()

    completed_features = completed_before.groupby(
        "customer_id"
    ).agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    ).reset_index()

    completed_features["recency"] = (
        CUTOFF_DATE - completed_features["last_order_date"]
    ).dt.days

    favorite_category = (
        completed_before.groupby("customer_id")["category"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="favorite_category")
    )

    order_counts = before_cutoff.groupby(
        "customer_id"
    ).agg(
        total_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda x: (x == "cancelled").sum(),
        ),
    ).reset_index()

    order_counts["cancellation_rate"] = (
        order_counts["cancelled_orders"]
        / order_counts["total_orders"]
    )

    dataset = pd.DataFrame({
        "customer_id": cohort_ids
    })

    dataset = dataset.merge(
        completed_features,
        on="customer_id",
        how="left",
    )

    dataset = dataset.merge(
        favorite_category,
        on="customer_id",
        how="left",
    )

    dataset = dataset.merge(
        order_counts[
            ["customer_id", "cancellation_rate"]
        ],
        on="customer_id",
        how="left",
    )

    dataset = dataset.merge(
        customers[
            [
                "customer_id",
                "city",
                "age",
                "gender",
                "signup_date",
            ]
        ],
        on="customer_id",
        how="left",
    )

    dataset["tenure_days"] = (
        CUTOFF_DATE - dataset["signup_date"]
    ).dt.days

    september_completed = orders[
        (orders["order_ts"] >= CUTOFF_DATE)
        & (orders["order_ts"] < LABEL_END_DATE)
        & (orders["status"] == "completed")
    ]

    active_september_ids = set(
        september_completed["customer_id"]
    )

    dataset["lapsed"] = (
        ~dataset["customer_id"].isin(active_september_ids)
    ).astype(int)

    dataset = dataset[[
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favorite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed",
    ]]

    feature_columns = dataset.drop(
        columns=["customer_id", "lapsed"]
    )

    missing_values = int(
        feature_columns.isna().sum().sum()
    )

    print("\nMissing feature values:", missing_values)

    assert missing_values == 0, (
        "Missing values found in lapse-model features."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(
        OUTPUT_DIR / "lapse_dataset.csv",
        index=False,
    )

    print("\nCohort size:", len(dataset))
    print("Lapsed customers:", dataset["lapsed"].sum())
    print(
        "Lapsed percentage:",
        round(dataset["lapsed"].mean() * 100, 2),
        "%",
    )

    X = dataset.drop(
        columns=["customer_id", "lapsed"]
    )

    y = dataset["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    print("\nMajority-Class Baseline:")
    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, baseline_predictions) * 100,
            2,
        ),
        "%",
    )
    print(
        "Lapsed F1:",
        round(
            f1_score(
                y_test,
                baseline_predictions,
                zero_division=0,
            ),
            3,
        ),
    )


def train_lapse_model():
    global _MODEL_CONTEXT

    data = pd.read_csv(
        OUTPUT_DIR / "lapse_dataset.csv"
    )

    X = data.drop(
        columns=["customer_id", "lapsed"]
    )

    y = data["lapsed"]

    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]

    categorical_features = [
        "favorite_category",
        "city",
        "gender",
    ]

    assert not X.isna().any().any(), (
        "Missing values found before model preprocessing."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train_numeric = scaler.fit_transform(
        X_train[numeric_features]
    )

    X_test_numeric = scaler.transform(
        X_test[numeric_features]
    )

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
    )

    X_train_categorical = encoder.fit_transform(
        X_train[categorical_features]
    )

    X_test_categorical = encoder.transform(
        X_test[categorical_features]
    )

    full_cohort_mean = X["frequency"].mean()

    frequency_index = numeric_features.index(
        "frequency"
    )

    training_mean = scaler.mean_[frequency_index]

    print(
        "\nFull cohort frequency mean:",
        round(full_cohort_mean, 2),
    )

    print(
        "Training frequency mean:",
        round(training_mean, 2),
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Numerical training shape:", X_train_numeric.shape)
    print("Numerical testing shape:", X_test_numeric.shape)
    print("Categorical training shape:", X_train_categorical.shape)
    print("Categorical testing shape:", X_test_categorical.shape)

    print(
        "\nAll required feature columns present:",
        set(
            numeric_features + categorical_features
        ).issubset(X.columns),
    )

    print("\nPreprocessing completed successfully.")

    X_train_processed = np.hstack([
        X_train_numeric,
        X_train_categorical.toarray(),
    ])

    X_test_processed = np.hstack([
        X_test_numeric,
        X_test_categorical.toarray(),
    ])

    logistic = LogisticRegression(max_iter=1000)

    logistic_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }

    logistic_search = GridSearchCV(
        logistic,
        logistic_grid,
        cv=5,
        scoring="f1",
    )

    logistic_search.fit(
        X_train_processed,
        y_train,
    )

    best_logistic = logistic_search.best_estimator_

    print("\nLogistic Regression")
    print("Best parameters:", logistic_search.best_params_)
    print("Best CV F1:", round(logistic_search.best_score_, 3))

    logistic_predictions = best_logistic.predict(
        X_test_processed
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, logistic_predictions))
    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    rf_grid = {
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [5, 10, 20],
    }

    rf_search = GridSearchCV(
        random_forest,
        rf_grid,
        cv=5,
        scoring="f1",
    )

    rf_search.fit(
        X_train_processed,
        y_train,
    )

    best_rf = rf_search.best_estimator_

    print("\nRandom Forest")
    print("Best parameters:", rf_search.best_params_)
    print("Best CV F1:", round(rf_search.best_score_, 3))

    rf_predictions = best_rf.predict(
        X_test_processed
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                rf_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                rf_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                rf_predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    logistic_f1 = f1_score(
        y_test,
        logistic_predictions,
        zero_division=0,
    )

    random_forest_f1 = f1_score(
        y_test,
        rf_predictions,
        zero_division=0,
    )

    print("\nModel Comparison")
    print("Logistic Regression F1:", round(logistic_f1, 3))
    print("Random Forest F1:", round(random_forest_f1, 3))

    if random_forest_f1 > logistic_f1:
        best_model = best_rf
        best_model_name = "Random Forest"
        selected_predictions = rf_predictions
    else:
        best_model = best_logistic
        best_model_name = "Logistic Regression"
        selected_predictions = logistic_predictions

    print("\nSelected Model:", best_model_name)

    X_majority = X_train_processed[
        y_train.values == 0
    ]

    X_minority = X_train_processed[
        y_train.values == 1
    ]

    y_minority = y_train[
        y_train == 1
    ]

    additional_samples = (
        len(X_majority) - len(X_minority)
    )

    if additional_samples > 0:
        X_minority_extra, y_minority_extra = resample(
            X_minority,
            y_minority,
            replace=True,
            n_samples=additional_samples,
            random_state=42,
        )

        X_train_balanced = np.vstack([
            X_train_processed,
            X_minority_extra,
        ])

        y_train_balanced = np.concatenate([
            y_train.values,
            y_minority_extra.values,
        ])
    else:
        X_train_balanced = X_train_processed.copy()
        y_train_balanced = y_train.values.copy()

    print("\nOriginal training rows:", len(X_train_processed))
    print("Additional minority rows:", additional_samples)
    print("Balanced training rows:", len(X_train_balanced))
    print("\nBalanced class distribution:")
    print(pd.Series(y_train_balanced).value_counts())

    balanced_model = LogisticRegression(
        C=logistic_search.best_params_["C"],
        max_iter=1000,
    )

    balanced_model.fit(
        X_train_balanced,
        y_train_balanced,
    )

    balanced_predictions = balanced_model.predict(
        X_test_processed
    )

    print("\nOriginal Logistic Regression")
    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                logistic_predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    print("\nBalanced Logistic Regression")
    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                balanced_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                balanced_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                balanced_predictions,
                zero_division=0,
            ),
            3,
        ),
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, balanced_predictions))

    feature_names = numeric_features + list(
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": best_rf.feature_importances_,
    }).sort_values(
        by="importance",
        ascending=False,
    )

    print("\nTop 5 Feature Importances:")
    print(importance_df.head(5).to_string(index=False))

    coefficient_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": best_logistic.coef_[0],
    })

    coefficient_df["direction"] = (
        coefficient_df["coefficient"].apply(
            lambda x: (
                "Increases lapse likelihood"
                if x > 0
                else "Decreases lapse likelihood"
            )
        )
    )

    coefficient_df["abs_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        by="abs_coefficient",
        ascending=False,
    )

    print("\nTop 3 Logistic Regression Coefficients:")
    print(
        coefficient_df[
            ["feature", "coefficient", "direction"]
        ].head(3).to_string(index=False)
    )

    model_file = OUTPUT_DIR / "lapse_model.joblib"

    joblib.dump(best_model, model_file)

    loaded_model = joblib.load(model_file)

    reloaded_predictions = loaded_model.predict(
        X_test_processed
    )

    predictions_match = np.array_equal(
        reloaded_predictions,
        selected_predictions,
    )

    print("\nSelected model predictions match:", predictions_match)

    assert predictions_match, (
        "Reloaded model predictions do not match."
    )

    segments = pd.read_csv(
        OUTPUT_DIR / "customer_segments.csv"
    )

    premium_ids = segments.loc[
        segments["is_premium"] == True,
        "customer_id",
    ]

    _MODEL_CONTEXT = {
        "data": data,
        "X_test": X_test,
        "X_test_processed": X_test_processed,
        "y_test": y_test,
        "loaded_model": loaded_model,
        "premium_ids": premium_ids,
        "selected_model_name": best_model_name,
    }

    return _MODEL_CONTEXT


def create_retention_list():
    if _MODEL_CONTEXT is None:
        raise RuntimeError(
            "Run train_lapse_model() before create_retention_list()."
        )

    context = _MODEL_CONTEXT

    data = context["data"]
    X_test = context["X_test"]
    X_test_processed = context["X_test_processed"]
    loaded_model = context["loaded_model"]
    premium_ids = context["premium_ids"]

    test_customers = data.loc[
        X_test.index,
        ["customer_id", "lapsed"],
    ].copy()

    test_customers["lapse_probability"] = (
        loaded_model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    test_customers["is_premium"] = (
        test_customers["customer_id"].isin(premium_ids)
    )

    retention_list = test_customers[
        test_customers["is_premium"]
    ].copy()

    retention_list = retention_list.sort_values(
        by="lapse_probability",
        ascending=False,
    )

    retention_list = retention_list.rename(
        columns={
            "lapsed": "actually_lapsed"
        }
    )

    retention_list = retention_list[[
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]]

    retention_file = OUTPUT_DIR / "retention_list.csv"

    retention_list.head(10).to_csv(
        retention_file,
        index=False,
    )

    print("\nTop 10 Premium Customers for Retention:")
    print(retention_list.head(10).to_string(index=False))

    print(
        "\nActually lapsed among top 10:",
        retention_list.head(10)["actually_lapsed"].sum(),
    )

    print("\nRetention list saved:", retention_file)

    return retention_list.head(10)


if __name__ == "__main__":
    rfm_and_segments()
    build_lapse_dataset()
    train_lapse_model()
    create_retention_list()
