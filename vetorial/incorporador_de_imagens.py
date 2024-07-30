import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode

import chromadb as db

MODELO = "/misc/workspace/inteligencia artificial/embeddings/modelos/mobilenet_v3_large.tflite"
BANCO = "/misc/workspace/inteligencia artificial/vetorial/banco"

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


def configurar():
    configurado, incorporador, conexao_banco = False, None, None

    try:
        opcoes = ImageEmbedderOptions(base_options=BaseOptions(model_asset_path=MODELO), running_mode=RunningMode.IMAGE)
        incorporador = ImageEmbedder.create_from_options(opcoes)

        conexao_banco = db.PersistentClient(path=BANCO)

        configurado = True
    except Exception as e:
        print(f"erro configurando incorporador: {str(e)}")

    return configurado, incorporador, conexao_banco

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

def converter_incorporacao(incorporacao):
    conversao = []

    for valor in incorporacao:
        conversao.append(int(valor))

    return conversao

def gravar_incorporacoes(grupo_de_bichos, incorporacoes_do_grupo, conexao_banco):
    gravado = False

    try:
        colecao = conexao_banco.get_or_create_collection(grupo_de_bichos)

        for id, incorporacao in enumerate(incorporacoes_do_grupo):
            colecao.add(embeddings=[converter_incorporacao(incorporacao.embeddings[0].embedding)], ids = [str(id + 1)])

        gravado = True
    except Exception as e:
        print(f"ocorreu um erro gravando as incorporações: {str(e)}")

    return gravado

if __name__ == "__main__":
    configurado, incorporador, conexao_banco = configurar()
    if configurado:
        _, vetores_de_gatos_pretos = processar_bichos(GATOS_PRETOS, incorporador)
        gravar_incorporacoes("gatos_pretos", vetores_de_gatos_pretos, conexao_banco)

        _, vetores_de_gatos_brancos = processar_bichos(GATOS_BRANCOS, incorporador)
        gravar_incorporacoes("gatos_brancos", vetores_de_gatos_brancos, conexao_banco)

        _, vetores_de_cachorros = processar_bichos(CACHORROS, incorporador)
        gravar_incorporacoes("cachorros", vetores_de_cachorros, conexao_banco)