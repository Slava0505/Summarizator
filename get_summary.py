import fire
from summarizators.luhn_summarizator import LuhnSummarizator
from tools.lang_identifying import lang_identify

def logging(log):
    pass

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
    print(f'Text language is {lang}')
    logging(f'Text language is {lang}')

    summed_text = summarizator.sum_text(text)
    input_file_name = input_file_path.split('/')[-1]
    out_file = open(out_dir+'summed_'+input_file_name, 'w', encoding='utf-8')
    out_file.write(summed_text)
    out_file.close()

if __name__ == '__main__':
    fire.Fire(get_summary)
