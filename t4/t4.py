import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

def plot_bar(data, figname, bar_width, xlabel = None, ylabel = None, title = None, invert_yaxis = False, horizontal = False):
    """ Procedimento genérico para plotar um gráfico de barras (horizontal ou vertical) com base nos dados de entrada. """
    fig, ax = plt.subplots(figsize = (10, 5))
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
        ax.set(xticks = indices, xticklabels = data['labels'])

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

def plot_quantidade_por_min_support(df, first, last):
    """ Plote a quantidade de itemsets para valores de min_support (%) no intervalo [first, last] """
    min_supports = []
    quantidade = []

    # Testando min_support de 0.01 a 0.20
    for n in range(first, last + 1):
        itemsets = apriori(df, min_support = n/100.0, use_colnames = True)
        min_supports.append(n)
        quantidade.append(len(itemsets))

    data = {
        'labels': min_supports,
        'bars': [
            {
                'heights': quantidade,
                'label': None,
                'color': 'darkblue'
            }
        ]
    }

    plot_bar(
        data, 
        figname = 'qt_min_support.png',
        xlabel = 'min_support (%)', 
        ylabel = 'Quantidade', 
        bar_width = 0.6, 
        title = None
    )

df = pd.read_csv('trabalho4_dados_01.csv')

plot_quantidade_por_min_support(df, 1, 20)

itemsets = apriori(df, min_support = 0.01, use_colnames = True)
regras = association_rules(itemsets, metric = "confidence", min_threshold = 0.95)
regras.sort_values(by = "lift", ascending = False, inplace = True)

print(df)
print(itemsets)
print(regras)
