import random
import re
from collections import defaultdict
from typing import List, Tuple, Optional

class NGramLanguageModel:
    """
    Генеративная языковая модель на основе N-грамм и цепей Маркова.
    Поддерживает произвольный размер контекста (N-grams).
    """
    def __init__(self, n: int = 2):
        if n < 2:
            raise ValueError("Размер N-граммы должен быть не менее 2 (биграммы).")
        self.n: int = n
        # Ключ: контекст (кортеж из N-1 слов), Значение: список возможных следующих слов
        self.model: defaultdict = defaultdict(list)
        
    def _tokenize(self, text: str) -> List[str]:
        """Очистка текста и приведение к нижнему регистру."""
        return re.findall(r'\w+', text.lower())

    def fit(self, text: str) -> 'NGramLanguageModel':
        """
        Обучение модели: построение матрицы переходных вероятностей.
        """
        words = self._tokenize(text)
        if len(words) < self.n:
            raise ValueError(f"Длина текста ({len(words)}) меньше размера N-граммы ({self.n}).")

        # Проход по тексту скользящим окном размера N
        for i in range(len(words) - self.n + 1):
            ngram = words[i:i + self.n]
            context = tuple(ngram[:-1])
            next_word = ngram[-1]
            self.model[context].append(next_word)
            
        return self

    def generate(self, length: int = 100, seed: Optional[Tuple[str, ...]] = None) -> str:
        """
        Генерация последовательности текста заданной длины.
        """
        if not self.model:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод .fit().")

        # Выбор стартового контекста
        context = seed if seed in self.model else random.choice(list(self.model.keys()))
        generated_words = list(context)

        for _ in range(length - len(context)):
            current_context = tuple(generated_words[-(self.n - 1):])
            
            # Обработка тупикового контекста (если слово было только в самом конце текста)
            if current_context not in self.model:
                current_context = random.choice(list(self.model.keys()))
                generated_words.extend(list(current_context))
                if len(generated_words) >= length:
                    break

            next_word = random.choice(self.model[current_context])
            generated_words.append(next_word)

        return ' '.join(generated_words[:length])


def load_text(file_name: str) -> str:
    """Чтение файла с обработкой исключений."""
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_name}' не найден.")
        return ""


if __name__ == "__main__":
    raw_text = load_text('text.txt')
    
    if raw_text:
        # Инициализация и обучение биграммной модели
        bigram_model = NGramLanguageModel(n=2)
        bigram_model.fit(raw_text)
        generated_bi = bigram_model.generate(length=100)

        # Инициализация и обучение триграммной модели
        trigram_model = NGramLanguageModel(n=3)
        trigram_model.fit(raw_text)
        generated_tri = trigram_model.generate(length=300)

        # Логирование результатов
        try:
            with open('generated_text.txt', 'w', encoding='utf-8') as file:
                file.write(f"=== Generated Bigram Text (N=2) ===\n{generated_bi}\n\n")
                file.write(f"=== Generated Trigram Text (N=3) ===\n{generated_tri}\n")
            print("Текст успешно сгенерирован и сохранен в 'generated_text.txt'")
        except IOError as e:
            print(f"Ошибка записи в файл: {e}")