import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


st.set_page_config(
    page_title="CNSL Production Forecast",
    page_icon="🌰",
    layout="wide"
)

st.title("🌰 CNSL Production Forecast using LSTM")
st.write("Cashew Nut Shell Liquid Quantity Prediction and Forecasting")


uploaded_file = st.file_uploader(
    "Upload Cashew CSV Dataset",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    df["Quantity (in M.T.)"] = pd.to_numeric(
        df["Quantity (in M.T.)"],
        errors="coerce"
    )

    df["Value (in Rs Crore)"] = pd.to_numeric(
        df["Value (in Rs Crore)"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Year", "Quantity (in M.T.)"]
    )

    yearly_data = (
        df.groupby("Year")["Quantity (in M.T.)"]
        .sum()
        .reset_index()
    )

    st.subheader("Annual CNSL Quantity")

    st.dataframe(yearly_data)

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        yearly_data["Year"],
        yearly_data["Quantity (in M.T.)"],
        marker="o"
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("CNSL Quantity (M.T.)")
    ax1.set_title("Annual CNSL Quantity")
    ax1.grid(True)

    st.pyplot(fig1)


    if len(yearly_data) < 20:

        st.error(
            "Dataset should contain at least 20 yearly observations."
        )

    else:

        st.subheader("LSTM Model Settings")

        train_size = 17
        time_steps = 3

        data = yearly_data[
            "Quantity (in M.T.)"
        ].values.reshape(-1, 1)

        scaler = MinMaxScaler()

        train_raw = data[:train_size]

        scaler.fit(train_raw)

        train_scaled = scaler.transform(train_raw)

        test_raw = data[train_size - time_steps:]

        test_scaled = scaler.transform(test_raw)


        def create_sequences(data, time_steps):

            X = []
            y = []

            for i in range(len(data) - time_steps):

                X.append(
                    data[i:i + time_steps]
                )

                y.append(
                    data[i + time_steps]
                )

            return np.array(X), np.array(y)


        X_train, y_train = create_sequences(
            train_scaled,
            time_steps
        )

        X_test, y_test = create_sequences(
            test_scaled,
            time_steps
        )


        st.write(
            "Training samples:",
            X_train.shape[0]
        )

        st.write(
            "Testing samples:",
            X_test.shape[0]
        )


        model = Sequential([
            Input(shape=(time_steps, 1)),
            LSTM(16),
            Dropout(0.1),
            Dense(8, activation="relu"),
            Dense(1)
        ])


        model.compile(
            optimizer="adam",
            loss="mse"
        )


        early_stop = EarlyStopping(
            monitor="loss",
            patience=30,
            restore_best_weights=True
        )


        with st.spinner("Training LSTM model..."):

            history = model.fit(
                X_train,
                y_train,
                epochs=500,
                batch_size=2,
                callbacks=[early_stop],
                verbose=0
            )


        st.success("LSTM model trained successfully!")


        st.subheader("Training Loss")

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.plot(
            history.history["loss"]
        )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("LSTM Training Loss")
        ax2.grid(True)

        st.pyplot(fig2)


        pred_scaled = model.predict(
            X_test,
            verbose=0
        )

        pred_quantity = scaler.inverse_transform(
            pred_scaled
        ).flatten()

        actual_quantity = scaler.inverse_transform(
            y_test
        ).flatten()


        test_years = yearly_data[
            "Year"
        ].iloc[train_size:].values


        results = pd.DataFrame({
            "Year": test_years,
            "Actual Quantity": actual_quantity,
            "Predicted Quantity": pred_quantity
        })


        st.subheader("Actual vs Predicted")

        st.dataframe(
            results.style.format({
                "Actual Quantity": "{:.2f}",
                "Predicted Quantity": "{:.2f}"
            })
        )


        mae = mean_absolute_error(
            actual_quantity,
            pred_quantity
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_quantity,
                pred_quantity
            )
        )

        r2 = r2_score(
            actual_quantity,
            pred_quantity
        )

        smape = np.mean(
            2 * np.abs(
                pred_quantity - actual_quantity
            ) /
            (
                np.abs(actual_quantity)
                + np.abs(pred_quantity)
                + 1e-8
            )
        ) * 100


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MAE",
            f"{mae:.2f}"
        )

        col2.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        col3.metric(
            "R² Score",
            f"{r2:.4f}"
        )

        col4.metric(
            "sMAPE",
            f"{smape:.2f}%"
        )


        st.subheader(
            "Actual vs Predicted CNSL Quantity"
        )

        fig3, ax3 = plt.subplots(
            figsize=(12, 6)
        )

        ax3.plot(
            results["Year"],
            results["Actual Quantity"],
            marker="o",
            label="Actual"
        )

        ax3.plot(
            results["Year"],
            results["Predicted Quantity"],
            marker="o",
            label="Predicted"
        )

        ax3.set_xlabel("Year")
        ax3.set_ylabel("Quantity (M.T.)")
        ax3.set_title(
            "LSTM: Actual vs Predicted CNSL Quantity"
        )

        ax3.legend()
        ax3.grid(True)

        st.pyplot(fig3)


        st.subheader(
            "🔮 Future CNSL Quantity Forecast"
        )


        forecast_years = st.slider(
            "Select number of future years",
            min_value=1,
            max_value=10,
            value=5
        )


        all_scaled = scaler.transform(data)

        last_sequence = all_scaled[
            -time_steps:
        ].copy()

        future_predictions = []


        for i in range(forecast_years):

            next_prediction = model.predict(
                last_sequence.reshape(
                    1,
                    time_steps,
                    1
                ),
                verbose=0
            )

            future_predictions.append(
                next_prediction[0, 0]
            )

            last_sequence = np.append(
                last_sequence[1:],
                next_prediction
            ).reshape(
                time_steps,
                1
            )


        future_predictions = scaler.inverse_transform(
            np.array(
                future_predictions
            ).reshape(-1, 1)
        ).flatten()


        last_year = int(
            yearly_data["Year"].iloc[-1]
        )

        future_years = np.arange(
            last_year + 1,
            last_year + forecast_years + 1
        )


        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Forecasted Quantity (M.T.)":
                future_predictions
        })


        st.dataframe(
            forecast_df.style.format({
                "Forecasted Quantity (M.T.)":
                    "{:.2f}"
            })
        )


        fig4, ax4 = plt.subplots(
            figsize=(12, 6)
        )

        ax4.plot(
            yearly_data["Year"],
            yearly_data["Quantity (in M.T.)"],
            marker="o",
            label="Historical Data"
        )

        ax4.plot(
            future_years,
            future_predictions,
            marker="o",
            linestyle="--",
            label="LSTM Forecast"
        )

        ax4.set_xlabel("Year")
        ax4.set_ylabel("Quantity (M.T.)")
        ax4.set_title(
            "CNSL Quantity Forecast using LSTM"
        )

        ax4.legend()
        ax4.grid(True)

        st.pyplot(fig4)


        st.subheader("📥 Download Forecast")

        csv = forecast_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="cnsl_lstm_forecast.csv",
            mime="text/csv"
        )