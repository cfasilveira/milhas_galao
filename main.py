import pandas as pd
import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('milhas_galao.csv')
df.info()
df['kilo milhares'] = df['Libras em milhares '] * 0.453592
print(tabulate.tabulate(df, headers='keys', tablefmt='fancy_grid'))

# Graficos comparativos das fetures com print(tabulate.tabulate(df, headers='keys', tablefmt='fancy_grid'))
# subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=('Cilindrada/Milhas', 'Cilindros/Milhas', 'CV/Milhas', 'Kilos/Milhas'))

# loop plotagem
row = 1
col = 1
for feature in ['Cilindrada', 'Cilindros', 'CV', 'kilo milhares']:
    fig.add_trace(go.Scatter(x=df[feature], y=df['Milhas por galao'], mode='markers', name=f'{feature} vs Milhas'), row=row, col=col)
    # atualizar titulos
    fig.update_xaxes(title_text=feature, row=row, col=col)
    fig.update_yaxes(title_text='Milhas por galão', row=row, col=col)
    # atualizar linha e coluna
    col += 1
    if col > 2:
        col = 1
        row += 1

# atualizar layout
fig.update_layout(title='Relação entre diferentes características e Milhas por galão', showlegend=False)
fig.show()

# Matriz de correlacao
corr = df.corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title='Heatmap da Matriz de Correlação')
fig_corr.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
fig_corr.show()

