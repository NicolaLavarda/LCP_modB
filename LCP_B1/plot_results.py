import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Percorso del file CSV
    file_path = "results.csv"
    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_path}' non esiste nella cartella corrente.")
        return
    
    # Lettura dei dati
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        return
        
    print(f"Letti {len(df)} risultati dal file {file_path}")
    
    # Crea cartella di output
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Imposta lo stile di Seaborn
    sns.set_theme(style="whitegrid", context="talk")
    
    # ------------------------------------------------------------------------
    # 1. GRAFICI GLOBALI SUI RISULTATI (Andamenti al variare dei parametri)
    # ------------------------------------------------------------------------
    
    # Convertiamo il learning rate in stringa formattata per usarlo come categoria ordinata nei boxplot
    df['lr_cat'] = df['learning_rate'].apply(lambda x: f"{x:g}")
    lr_order = sorted(df['lr_cat'].unique(), key=float)
    
    # A) Boxplots per valutare la distribuzione delle performance per ciascun parametro
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribuzione della Validation Accuracy per singolo Iperparametro', fontsize=20, y=1.02)
    
    # Boxplot Learning Rate
    sns.boxplot(ax=axs[0, 0], data=df, x='lr_cat', y='val_accuracy', order=lr_order, palette="viridis")
    axs[0, 0].set_title('Validation Acc vs Learning Rate')
    axs[0, 0].set_xlabel('Learning Rate')
    axs[0, 0].set_ylabel('Validation Accuracy')
    
    # Boxplot Activation
    sns.boxplot(ax=axs[0, 1], data=df, x='activation', y='val_accuracy', palette="viridis")
    axs[0, 1].set_title('Validation Acc vs Activation Function')
    axs[0, 1].set_xlabel('Activation')
    axs[0, 1].set_ylabel('Validation Accuracy')

    # Boxplot Dropout
    sns.boxplot(ax=axs[1, 0], data=df, x='dropout', y='val_accuracy', palette="viridis")
    axs[1, 0].set_title('Validation Acc vs Dropout')
    axs[1, 0].set_xlabel('Dropout')
    axs[1, 0].set_ylabel('Validation Accuracy')
    
    # Boxplot Optimizer
    sns.boxplot(ax=axs[1, 1], data=df, x='optimizer', y='val_accuracy', palette="viridis")
    axs[1, 1].set_title('Validation Acc vs Optimizer')
    axs[1, 1].set_xlabel('Optimizer')
    axs[1, 1].set_ylabel('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_hyperparams_boxplots.png'), bbox_inches='tight')
    plt.close()

    # B) Interaction Plot: Validation Accuracy in funzione di Learning Rate per Optimizer
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='learning_rate', y='val_accuracy', hue='optimizer', style='optimizer', markers=True, dashes=False, palette="Set1")
    plt.xscale('log')
    plt.title('Validation Accuracy Media al variare di Learning Rate per Optimizer', fontsize=16)
    plt.xlabel('Learning Rate (Scala Log)')
    plt.ylabel('Validation Accuracy')
    plt.legend(title='Optimizer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_val_acc_by_lr_optimizer.png'))
    plt.close()

    # C) Interaction Plot: Validation Accuracy in funzione di Learning Rate per Activation
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='learning_rate', y='val_accuracy', hue='activation', style='activation', markers=True, dashes=False, palette="Set2")
    plt.xscale('log')
    plt.title('Validation Accuracy Media al variare di Learning Rate per Activation', fontsize=16)
    plt.xlabel('Learning Rate (Scala Log)')
    plt.ylabel('Validation Accuracy')
    plt.legend(title='Activation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_val_acc_by_lr_activation.png'))
    plt.close()


    # ------------------------------------------------------------------------
    # 2. I 5 MIGLIORI MODELLI
    # ------------------------------------------------------------------------
    
    # Seleziona i 5 modelli con la validation accuracy migliore
    top5_df = df.sort_values(by='val_accuracy', ascending=False).head(5).copy()
    
    # Crea una stringa descrittiva per ogni configurazione
    top5_df['config'] = (
        top5_df['optimizer'].str.upper() + " | " + 
        top5_df['activation'].str.title() + " | " + 
        "lr=" + top5_df['lr_cat'] + " | " + 
        "drop=" + top5_df['dropout'].astype(str)
    )
    
    # Prepara i dati in formato "long" per graficare affiancati Test vs Validation
    top5_melt = top5_df.melt(
        id_vars=['config'],
        value_vars=['val_accuracy', 'test_accuracy'],
        var_name='Metric',
        value_name='Accuracy'
    )
    top5_melt['Metric'] = top5_melt['Metric'].replace({'val_accuracy': 'Validation', 'test_accuracy': 'Test'})
    
    # Plot a barre orizzontali
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=top5_melt, y='config', x='Accuracy', hue='Metric', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Top 5 Model Configurations: Val vs Test Accuracy', fontsize=16)
    plt.xlabel('Accuracy')
    plt.ylabel('Configurazione (Optimizer | Activation | LR | Dropout)')
    
    # Imposta un limite minimo alle ascisse per dare risalto alle minime differenze, se i modelli sono molto simili
    min_acc = top5_melt['Accuracy'].min()
    plt.xlim(max(0, min_acc - 0.05), 1.0)
    
    # Aggiungi i valori textuali sopra le barre
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.002, p.get_y() + p.get_height()/2, f'{width:.4f}', 
                    va='center', fontsize=11)
            
    plt.legend(title='Split', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_top_5_models.png'))
    plt.close()

    print("\nSuccesso! Sono stati generati i seguenti grafici nella cartella 'plots/':")
    print(" 1) 1_hyperparams_boxplots.png   - Boxplots per ogni parametro")
    print(" 2) 2_val_acc_by_lr_optimizer.png- Trend accuracy vs LR (distinto per Optimizer)")
    print(" 3) 3_val_acc_by_lr_activation.png- Trend accuracy vs LR (distinto per Activation)")
    print(" 4) 4_top_5_models.png           - Bar chart con Val e Test Accuracy per le 5 configurazioni migliori")
    print("\nEcco il dettaglio in forma tabellare dei top 5:")
    print(top5_df[['config', 'val_accuracy', 'test_accuracy']].to_string(index=False))

    # ------------------------------------------------------------------------
    # 3. GRAFICI 2D PER I TOP 5 MODELLI (variando un parametro alla volta)
    # ------------------------------------------------------------------------
    
    # Lista dei parametri che vogliamo esplorare
    params_to_plot = ['learning_rate', 'dropout', 'activation', 'optimizer']
    
    for i, (_, row) in enumerate(top5_df.iterrows()):
        model_rank = i + 1
        
        # Estraiamo la configurazione base di questo top model
        base_lr = row['learning_rate']
        base_drop = row['dropout']
        base_act = row['activation']
        base_opt = row['optimizer']
        
        # Prepariamo la figura con 4 subplots (uno per ogni parametro)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Top {model_rank} Model Analysis | Base: {row['config']} | Val Acc: {row['val_accuracy']:.4f}", 
                     fontsize=20, y=1.05)
        
        for j, param in enumerate(params_to_plot):
            # Filtriamo il dataframe: teniamo costanti tutti gli ALTRI parametri
            # e lasciamo variare solo 'param'
            mask = pd.Series(True, index=df.index)
            if param != 'learning_rate': mask &= (df['learning_rate'] == base_lr)
            if param != 'dropout':       mask &= (df['dropout'] == base_drop)
            if param != 'activation':    mask &= (df['activation'] == base_act)
            if param != 'optimizer':     mask &= (df['optimizer'] == base_opt)
            
            df_plot = df[mask].copy()
            
            # Se ci sono dati da mostrare (almeno due punti per fare un senso di trade-off)
            if len(df_plot) > 0:
                # Ordina per il parametro corrente per avere un plot coerente
                df_plot = df_plot.sort_values(by=param)
                
                # Creiamo il plot in base al tipo di parametro (categorico vs numerico)
                if param == 'learning_rate':
                    sns.lineplot(ax=axs[j], data=df_plot, x=param, y='val_accuracy', marker='o', label='Val Acc')
                    sns.lineplot(ax=axs[j], data=df_plot, x=param, y='test_accuracy', marker='s', label='Test Acc', linestyle='--')
                    axs[j].set_xscale('log')
                else:
                    sns.lineplot(ax=axs[j], data=df_plot, x=param, y='val_accuracy', marker='o', label='Val Acc')
                    sns.lineplot(ax=axs[j], data=df_plot, x=param, y='test_accuracy', marker='s', label='Test Acc', linestyle='--')
                
                # Aggiungiamo un indicatore (asterisco rosso) per il valore "Base" di questo top model
                base_val = row[param]
                base_acc = row['val_accuracy']
                axs[j].scatter([base_val], [base_acc], color='red', s=150, zorder=5, marker='*', label='Base Config')

                axs[j].set_title(f"Variando: {param.title()}")
                axs[j].set_xlabel(param.title())
                axs[j].set_ylabel('Accuracy')
                axs[j].legend()
                
                # Fissiamo i limiti dell'asse y per avere consistenza
                axs[j].set_ylim(max(0, df_plot['test_accuracy'].min() - 0.1), 1.0)
            else:
                axs[j].text(0.5, 0.5, "Dati non disponibili", ha='center', va='center')
                axs[j].set_title(f"Variando: {param.title()}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'5_top{model_rank}_model_analysis.png'), bbox_inches='tight')
        plt.close()

    print("\n Sono stati aggiunti 5 nuovi grafici (uno per ognuna delle top-5 configurazioni) denominati '5_topX_model_analysis.png'")


if __name__ == "__main__":
    main()
