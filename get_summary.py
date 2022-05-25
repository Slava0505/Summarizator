import fire
from summarizators.luhn_summarizator import LuhnSummarizator
from tools.lang_identifying import lang_identify
import logging

logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode="a")

def log_something(log):
    logging.info(log)

def get_summary(input_file_path='data/example_text_ru.txt', out_dir = 'data/', method = 'Luhn'):
    """
    Функцция получает саммари из содержимого
    входного файла и замисывает его в выходной

    Parameters
    ----------
    input_file_path:
    Путь к входному файлу

    out_dir:
    Дириктория для сохранения выходного
    файла с саммари

    method:
    Метод для суммаризации, на данный момент
    доступен только Luhn
    """
    if method == 'Luhn':
        summarizator = LuhnSummarizator()
    text = open(input_file_path, encoding='utf-8').read()

    lang = lang_identify(text)
    print(f'Язык текста: {lang}')
    log_something(f'Язык текста определен успешно: {lang}')

    summed_text = summarizator.sum_text(text)
    input_file_name = input_file_path.split('/')[-1]
    output_file_path = out_dir + 'summed_' + input_file_name
    out_file = open(output_file_path, 'w', encoding='utf-8')
    out_file.write(summed_text)
    out_file.close()
    log_something(f'Суммаризация звершена успешно. Создан файл {output_file_path}')

if __name__ == '__main__':
    fire.Fire(get_summary)
