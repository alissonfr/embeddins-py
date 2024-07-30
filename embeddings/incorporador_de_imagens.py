import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode

from colorama import Back, Style

MODELO = "/misc/workspace/inteligencia artificial/embeddings/modelos/mobilenet_v3_large.tflite"

GATOS_BRANCOS = [
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_branco1.png", 
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_branco2.png"
]

GATOS_PRETOS = [
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_preto1.png", 
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/gato_preto2.png"
]

CACHORROS = [
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/cachorro1.png", 
    "/misc/workspace/inteligencia artificial/embeddings/imagens_sem_fundo/cachorro2.png"
]

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


def configurar():
    configurado, incorporador = False, None

    try:
        opcoes = ImageEmbedderOptions(base_options=BaseOptions(model_asset_path=MODELO), running_mode=RunningMode.IMAGE)
        incorporador = ImageEmbedder.create_from_options(opcoes)

        configurado = True
    except Exception as e:
        print(f"erro configurando incorporador: {str(e)}")

    return configurado, incorporador

def processar(imagem, incorporador):
    processada, incorporacao = False, None

    try:
        imagem = mp.Image.create_from_file(imagem)
        incorporacao = incorporador.embed(imagem)

        processada = True
    except Exception as e:
        print(f"erro processando a imagem: {str(e)}")


    return processada, incorporacao

def processar_bichos(imagens_sem_fundo, incorporador):
    processados, incorporacoes = False, []

    for imagem in imagens_sem_fundo:
        processada, incorporacao = processar(imagem, incorporador)
        if processada:
            incorporacoes.append(incorporacao)

    processados = (len(incorporacoes) == len(imagens_sem_fundo))

    return processados, incorporacoes

def comparar(bicho, grupo_de_bichos, incorporador):
    similaridade = 0.0

    processada, incorporacao = processar(bicho['imagem'], incorporador)
    if processada:
        for bicho in grupo_de_bichos:
            comparacao = incorporador.cosine_similarity(incorporacao.embeddings[0], 
                                                        bicho.embeddings[0])
            similaridade = comparacao if comparacao > similaridade else similaridade
    else:
        print(f"não foi possível comparar o bicho {bicho['tipo']} através de sua imagem: {bicho['imagem']}")

    return similaridade

if __name__ == "__main__":
    configurado, incorporador = configurar()
    if configurado:
        _, vetores_de_gatos_pretos = processar_bichos(GATOS_PRETOS, incorporador)
        _, vetores_de_gatos_brancos = processar_bichos(GATOS_BRANCOS, incorporador)
        _, vetores_de_cachorros = processar_bichos(CACHORROS, incorporador)

        for bicho in BICHOS_PARA_TESTES:
            similaridade_com_gatos_pretos = comparar(bicho, vetores_de_gatos_pretos, incorporador)
            print(f"distancias do bicho {bicho['tipo']} a gatos pretos: {similaridade_com_gatos_pretos}")

            similaridade_com_gatos_brancos = comparar(bicho, vetores_de_gatos_brancos, incorporador)
            print(f"distancias do bicho {bicho['tipo']} a gatos brancos: {similaridade_com_gatos_brancos}")

            similaridade_com_cachorros = comparar(bicho, vetores_de_cachorros, incorporador)         
            print(f"distancias do bicho {bicho['tipo']} a cachorros: {similaridade_com_cachorros}")

            if similaridade_com_gatos_brancos > similaridade_com_gatos_pretos and similaridade_com_gatos_brancos > similaridade_com_cachorros:
                print(Back.CYAN + f"o bicho considerado {bicho['tipo']} é mais similar a gatos brancos" + Style.RESET_ALL)
            elif similaridade_com_gatos_pretos > similaridade_com_gatos_brancos and similaridade_com_gatos_pretos > similaridade_com_cachorros:
                print(Back.RED + f"o bicho considerado {bicho['tipo']} é mais similar a gatos pretos" + Style.RESET_ALL)
            else:
                print(Back.YELLOW + f"o bicho consideado {bicho['tipo']} é mais similar um cachorros" + Style.RESET_ALL)
