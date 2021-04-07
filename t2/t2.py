import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('trabalho2_dados_1.csv')

# Convert some columns to their expected data types
df['avaliacao-usuarios'] = pd.to_numeric(df['avaliacao-usuarios'], errors = 'coerce')
df['avaliacao-criticos'] = df['avaliacao-criticos'].div(10)
df['plataforma'] = df['plataforma'].astype('category')
df['genero'] = df['genero'].astype('category')

def plot_bar(figname, data, bar_width, xlabel = None, ylabel = None, title = None, invert_yaxis = False, horizontal = False):
    """ Procedimento genérico para plotar um gráfico de barras (horizontal ou vertical) com base nos dados de entrada. """
    fig, ax = plt.subplots(figsize = (16, 9))
    indices = np.arange(len(data['labels']))

    for i, bar in enumerate(data['bars']):        
        if horizontal:
            ax.barh(indices + i * bar_width, bar['heights'], color = bar['color'], label = bar['label'], height = bar_width)
        else:            
            ax.bar(indices + i * bar_width, bar['heights'], color = bar['color'], label = bar['label'], width = bar_width)

    ax.set_title(title)

    # Remove axes splines 
    for s in ['top', 'bottom', 'left', 'right']: 
        ax.spines[s].set_visible(False) 
    
    if horizontal:
        ax.set(yticks = indices, yticklabels = data['labels'])
    else:
        ax.set(xticks = indices + bar_width/len(data['bars']), xticklabels = data['labels'])

    # Remove x, y Ticks 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    # Add padding between axes and labels 
    ax.xaxis.set_tick_params(pad = 5) 
    ax.yaxis.set_tick_params(pad = 10) 

    # Set axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add x, y gridlines 
    ax.grid(b = True, color = 'grey', linestyle = '-.', linewidth = 0.5, alpha = 0.2) 

    if invert_yaxis:
        ax.invert_yaxis()
    
    # For each bar: place a label
    for rect in ax.patches:
        x_value = rect.get_width() if horizontal else rect.get_x() + rect.get_width()/2 
        y_value = rect.get_y() + rect.get_height()/2 if horizontal else rect.get_height()
        label = rect.get_width() if horizontal else rect.get_height()

        # Create annotation
        ax.annotate(
            str(round(label, 1)), 
            (x_value, y_value),       
            xytext = (5, 0) if horizontal else (0, 5),              
            textcoords = 'offset points', 
            va = 'center',                
            ha = 'left' if horizontal else 'center',
            fontsize = 10, 
            fontweight = 'bold', 
            color = 'grey'
        )                      

        ax.legend()
        fig.savefig(figname, bbox_inches = "tight")

def plot_top_x_jogos_vendas(x, figname = 'temp.png'):
    """ Plote os x jogos mais vendidos. """
    top_vendas_df = df.sort_values('vendas', ascending = False).head(x)

    data = {
        'labels': top_vendas_df['nome'],
        'bars': [
            {
                'heights': top_vendas_df['vendas'],
                'label': None,
                'color': 'm'
            }
        ]
    }

    plot_bar(
        figname, 
        data, 
        bar_width = 0.8, 
        title = 'Os {} jogos mais vendidos, em milhões de unidades'.format(x), 
        invert_yaxis = True, 
        horizontal = True
    )

def plot_top_x_fabricantes_vendas(x, figname = 'temp.png'):
    """ Plote as x fabricantes que mais venderam jogos (vendas acumuladas). """
    top_fabricantes_vendas_df = df.groupby('fabricante').sum().sort_values('vendas', ascending = False).head(x)

    data = {
        'labels': top_fabricantes_vendas_df.index.tolist(),
        'bars': [
            {
                'heights': top_fabricantes_vendas_df['vendas'],
                'label': None,
                'color': 'c'
            }
        ]
    }

    plot_bar(
        figname, 
        data, 
        bar_width = 0.8, 
        title = 'As {} fabricantes que mais venderam, em milhões de unidades'.format(x), 
        invert_yaxis = True, 
        horizontal = True
    )

def plot_plataforma_vendas(figname = 'temp.png'):
    """ Plote o total de jogos por plataforma e suas respectivas vendas acumuladas. """
    plataforma_vendas_df = df.groupby('plataforma')['vendas'].agg(total_vendas = 'sum', count = 'count')

    data = {
        'labels': plataforma_vendas_df.index.tolist(),
        'bars': [
            {
                'heights': plataforma_vendas_df['count'],
                'label': 'Total de jogos da plataforma na base de dados',
                'color': '#ff4554'
            },
            {
                'heights': plataforma_vendas_df['total_vendas'],
                'label': 'Vendas acumuladas, em milhões de unidades',
                'color': '#044b7d'
            },
        ] 
    }

    plot_bar(figname, data, bar_width = 0.4, title = 'Análise de jogos por plataforma')

def plot_genero_vendas(figname = 'temp.png'):
    """ Plote o total de jogos por gênero e suas respectivas vendas acumuladas. """
    genero_vendas_df = df.groupby('genero')['vendas'].agg(total_vendas = 'sum', count = 'count')

    data = {
        'labels': df['genero'].cat.categories,
        'bars': [
            {
                'heights': genero_vendas_df['count'],
                'label': 'Total de jogos do gênero na base de dados',
                'color': '#2a9df4'
            },
            {
                'heights': genero_vendas_df['total_vendas'],
                'label': 'Vendas acumuladas, em milhões de unidades',
                'color': '#044b7d'
            },
        ] 
    }

    plot_bar(figname, data, bar_width = 0.4, title = 'Análise de jogos por gênero')

def boxplot_avaliacao_usuarios_criticos(figname = 'temp.png'):
    """ Produz um boxplot contendo informações relacionadas às avaliações de usuários e de críticos. """
    total_usuarios = int(df['numero-usuarios'].sum())
    total_criticos = int(df['numero-criticos'].sum())

    fig, ax = plt.subplots(figsize = (16, 9))
    ax.set_title('Comparação entre a avaliação (nota) de usuários e críticos')

    boxplot = ax.boxplot(
        [df['avaliacao-usuarios'].dropna(), df['avaliacao-criticos'].dropna()],
        showmeans = True, 
        vert = True, 
        patch_artist = True, 
        labels = ['Usuários ({:,})'.format(total_usuarios), 'Críticos ({:,})'.format(total_criticos)]
    )

    # Fill with colors
    colors = ['pink', 'lightblue']

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Adjust yticks
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))

    # Adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_ylabel('Valores observados')

    fig.savefig(figname, bbox_inches = "tight")    

# Plotando os gráficos
plot_top_x_jogos_vendas(20, figname = 'img/top_20_jogos_vendas.png')
plot_top_x_fabricantes_vendas(20, figname = 'img/top_20_fabricantes_vendas.png')
plot_plataforma_vendas(figname = 'img/plataforma_vendas.png')
plot_genero_vendas(figname = 'img/genero_vendas.png')
boxplot_avaliacao_usuarios_criticos(figname = 'img/boxplot_usuarios_criticos.png')