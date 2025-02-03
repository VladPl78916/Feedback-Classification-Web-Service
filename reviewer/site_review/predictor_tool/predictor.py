from .ml_models import TextCNN
import torch
import dill
from site_review.vocab import Vocabulary


with open('site_review/predictor_tool/vocab_dl.dill', 'rb') as f:
    vocab = dill.load(f)

# Функция для загрузки модели
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TextCNN(vocab_size=len(vocab.vocabulary), embedding_dim=25,
                    pad_idx=vocab.get_pad())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Функция для предсказания
def predict_text(model, text, vocab, device, max_len=100):
    # Преобразуем текст в индексы
    encoded_text = vocab.encode(text)

    # Ограничиваем длину текста (или добавляем паддинг)
    if len(encoded_text) < max_len:
        encoded_text = encoded_text + \
            [vocab.get_pad()] * (max_len - len(encoded_text))

    # Преобразуем в тензор и добавляем размерность для батча
    encoded_text = torch.LongTensor(encoded_text).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(encoded_text)
        prediction = torch.round(torch.sigmoid(
            output)).item()  # Преобразуем в 0 или 1

    return prediction
