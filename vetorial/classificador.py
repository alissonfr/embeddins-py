from incorporador_de_imagens import *

from colorama import Back, Style


BICHOS_PARA_TESTES = [
    {
        "tipo": "cachorro",
        "imagem": "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/cachorro_teste.png"
    },
    {
        "tipo": "gato preto",
        "imagem": "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_preto_teste.png"
    },
    {
        "tipo": "gato branco",
        "imagem": "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_branco_teste.png"    
    }
]

def comparar(bicho, grupo_de_bichos, incorporador, conexao_banco):
    menor_distancia = float("inf")

    processada, incorporacao = processar(bicho['imagem'], incorporador)
    if processada:
        try:
            colecao = conexao_banco.get_collection(grupo_de_bichos)
            resultado = colecao.query(query_embeddings=[converter_incorporacao(incorporacao.embeddings[0].embedding)], n_results=2)

            distancias = resultado["distances"][0]
            for distancia in distancias:
                menor_distancia = distancia if distancia < menor_distancia else menor_distancia
        except Exception as e:
            print(f"ocorreu um erro durante a comparação: {str(e)}")
    else:
        print(f"não foi possível comparar o bicho {bicho['tipo']} através de sua imagem: {bicho['imagem']}")

    return distancia

if __name__ == "__main__":
    configurado, incorporador, conexao_banco = configurar()
    if configurado:
        for bicho in BICHOS_PARA_TESTES:
            distancia_de_gatos_pretos = comparar(bicho, "gatos_pretos", incorporador, conexao_banco)
            print(f"distancias do bicho {bicho['tipo']} a gatos pretos: {distancia_de_gatos_pretos}")

            distancia_de_gatos_brancos = comparar(bicho, "gatos_brancos", incorporador, conexao_banco)
            print(f"distancias do bicho {bicho['tipo']} a gatos brancos: {distancia_de_gatos_brancos}")

            distancia_de_cachorros = comparar(bicho, "cachorros", incorporador, conexao_banco)         
            print(f"distancias do bicho {bicho['tipo']} a cachorros: {distancia_de_cachorros}")

            if distancia_de_gatos_brancos < distancia_de_gatos_pretos and distancia_de_gatos_brancos < distancia_de_cachorros:
                print(Back.CYAN + f"o bicho considerado {bicho['tipo']} é mais similar a gatos brancos" + Style.RESET_ALL)
            elif distancia_de_gatos_pretos < distancia_de_gatos_brancos and distancia_de_gatos_pretos < distancia_de_cachorros:
                print(Back.RED + f"o bicho considerado {bicho['tipo']} é mais similar a gatos pretos" + Style.RESET_ALL)
            else:
                print(Back.YELLOW + f"o bicho consideado {bicho['tipo']} é mais similar um cachorros" + Style.RESET_ALL)

