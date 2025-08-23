import pandas as pd
import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('milhas_galao.csv')
df.info()
df['kilo milhares'] = df['Libras em milhares '] * 0.453592
print(tabulate.tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Padronização
scaler = StandardScaler()
df[['Milhas por galao', 'Cilindrada', 'Cilindros', 'CV', 'kilo milhares']] = scaler.fit_transform(df[['Milhas por galao', 'Cilindrada', 'Cilindros', 'CV', 'kilo milhares']])
print(tabulate.tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

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

# Correlações Positivas e Negativas
corr = df.iloc[:, 1:].corr() # Exclui a primeira coluna (Libras em milhares)
# Obter a parte superior triangular da matriz de correlação (excluindo a diagonal)
correlation_pairs = corr.stack().reset_index()
correlation_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Remover correlações da variável consigo mesma e pares duplicados
correlation_pairs = correlation_pairs[correlation_pairs['Variable 1'] != correlation_pairs['Variable 2']]
correlation_pairs = correlation_pairs.loc[
    correlation_pairs.apply(lambda x: ''.join(sorted((x['Variable 1'], x['Variable 2']))), axis=1).drop_duplicates().index
]

# Separar correlações positivas e negativas
positive_correlations = correlation_pairs[correlation_pairs['Correlation'] > 0].sort_values(by='Correlation', ascending=False)
negative_correlations = correlation_pairs[correlation_pairs['Correlation'] < 0].sort_values(by='Correlation', ascending=True)

print("Tabela de Correlações Positivas:")
print(tabulate.tabulate(positive_correlations, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\nTabela de Correlações Negativas:")
print(tabulate.tabulate(negative_correlations, headers='keys', tablefmt='fancy_grid', showindex=False))

# Loop de Regressao Linear com 'Milhas por galao' como variavel target, 
# com 'Kilos milhares' como uma feature fixa 
# e variando colunas 'Cilindrada', 'Cilindros', e 'CV' que sao correlativas
target_variable = 'Milhas por galao'
features_to_iterate = ['Cilindrada', 'Cilindros', 'CV']
fixed_feature = 'kilo milhares'

# loop de Regressao Linear
models = {} # armazena os resultados dos modelos
for feature in features_to_iterate:
    df_temp = df[[target_variable, fixed_feature, feature]].copy()

    X_temp = df_temp[[fixed_feature, feature]]
    y_temp = df_temp[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)

    models[feature] = {
        'model': model,
        'features': [fixed_feature, feature],
        'X_test': X_test,
        'y_test': y_test,
        
    }

print("Modelos trainados e armazenados com sucesso.")
print(models)

# Avaliacao dos modelos
r2_scores = {}

for feature, model_info in models.items():
    model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']

    r2 = model.score(X_test, y_test)
    r2_scores[feature] = r2

print("R² scores calculados e armazenados com sucesso.")
print(tabulate.tabulate(pd.DataFrame(r2_scores.items(), columns=['Feature', 'R² Score']), headers='keys', tablefmt='fancy_grid', showindex=False))

# Visualizacao performance dos modelos
cores = ['orange' if v < 0 else 'blue' for v in r2_scores.values()]
fig_perf = px.bar(
    x=list(r2_scores.keys()),
    y=list(r2_scores.values()),
    labels={'x': 'Feature', 'y': 'R² Score'},
    title='Performance dos Modelos',
    color=list(r2_scores.values()),
    color_discrete_map={v: 'orange' if v < 0 else 'blue' for v in r2_scores.values()}
)
fig_perf.update_traces(marker_color=cores)
fig_perf.show()

