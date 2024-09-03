from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usa un backend che non richiede una GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    avg_profit = 0
    max_drawdown = 0
    sharpe_ratio = 0
    df_combined_html = None
    graph_image = None

    if request.method == 'POST':
        # Recupera i dati dal form con .get() e valori predefiniti
        contracts = int(request.form.get('contracts', 1))
        min_ticks_profit = int(request.form.get('min_ticks_profit', 3))
        max_ticks_profit = int(request.form.get('max_ticks_profit', 7))
        ticks_loss = int(request.form.get('ticks_loss', 5))
        tick_value = float(request.form.get('tick_value', 12.5))
        fee_per_contract = float(request.form.get('fee_per_contract', 2.5))
        num_trades = int(request.form.get('num_trades', 200))
        breakeven_trades = int(request.form.get('breakeven_trades', 10))
        win_rate = int(request.form.get('win_rate', 60))
        num_variations = int(request.form.get('num_variations', 10))

        # Simulazione delle variations
        results = []
        for _ in range(num_variations):
            profits = []
            for _ in range(num_trades):
                if np.random.rand() < win_rate / 100:
                    profit = np.random.randint(min_ticks_profit, max_ticks_profit + 1) * tick_value
                else:
                    profit = -ticks_loss * tick_value
                profits.append(profit)
            cumulative_profit = np.cumsum(profits)
            results.append(cumulative_profit)

        # Combina i risultati in un unico DataFrame
        df_combined = pd.DataFrame(results).T
        df_combined.columns = [f'Variation {i+1}' for i in range(num_variations)]

        # Calcolo delle metriche
        avg_profit = df_combined.iloc[-1].mean()
        max_drawdown = df_combined.min().min()
        sharpe_ratio = avg_profit / df_combined.stack().std() if df_combined.stack().std() != 0 else 0
        sharpe_ratio = round(sharpe_ratio, 2)  # Arrotonda lo Sharpe Ratio a 2 decimali

        # Creazione del grafico con Matplotlib
        plt.figure(figsize=(10, 6))
        for col in df_combined.columns:
            plt.plot(df_combined.index, df_combined[col], label=col)
        plt.title('Cumulative Profit Chart for Variations')
        plt.xlabel('Trade')
        plt.ylabel('Cumulative Profit')
        plt.grid(True)
        plt.legend(loc='upper left')

        # Salva il grafico in un'immagine
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_image = base64.b64encode(img.getvalue()).decode()

        # Converte il dataframe in tabella HTML
        df_combined_html = df_combined.to_html(classes='table table-striped', index=False)

    return render_template('index.html', avg_profit=avg_profit, max_drawdown=max_drawdown,
                           sharpe_ratio=sharpe_ratio, df_combined_html=df_combined_html,
                           graph_image=graph_image)

if __name__ == '__main__':
    app.run(debug=True)
