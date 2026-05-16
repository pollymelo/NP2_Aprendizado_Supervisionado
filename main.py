"""Executa comparação de modelos, gera gráficos e salva tabela resumo."""

from pathlib import Path
from graficos import gerar_todos_graficos
from resultados import executar_comparacao


def main():
    comparacao, modelo_cv, tabela_bv = executar_comparacao()
    pasta = Path("graficos")
    pasta.mkdir(exist_ok=True)
    comparacao.to_csv(pasta / "tabela_comparacao_completa.csv", index=False)
    tabela_bv.to_csv(pasta / "tabela_bias_variance.csv", index=False)
    gerar_todos_graficos()
  
    print(f"\nTabela completa: {pasta / 'tabela_comparacao_completa.csv'}")
    print(f"Relatório (texto): RELATORIO_NP2.md")



if __name__ == "__main__":
    main()
