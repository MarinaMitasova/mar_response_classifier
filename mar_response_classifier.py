import time
import pandas as pd
from tqdm import trange
import time
import json
# import httpx
# from dotenv import load_dotenv
# import os
import openai
from openai import OpenAI
from tkinter.messagebox import askyesno


class ResponseClassifier:

    def __init__(self, tokkens_config: dict, prompts: dict, with_codekey: int, model_name: str, llm_type: str = 'OpenAI'):
        self.tokkens_config = tokkens_config
        self.prompts = prompts

        if with_codekey:
            self.sys_prompt = prompts['sysprompt_with_codekey']
        else:
            self.sys_prompt = prompts['sysprompt_without_codekey']

        self.llm_type = llm_type
        if model_name is None:
            model_name = self.tokkens_config[self.llm_type]['default_model']
        self.model_name = model_name
        self.openai_client = self.get_OpenAI_client()

    def json_to_pandas(self, json_result: list):
        df = pd.DataFrame(json_result)
        max_codes = 10
        for i in range(max_codes):
            df[f'code_{i + 1}'] = df['codes'].apply(
                lambda x: x[i] if i < len(x) else None)

        # Удаялем исходные коды
        df = df.drop(columns=['codes'])
        return df

    def json_prompt(self, prompt):
        # Добавляем инфомарцию, что ответ точно должен быть в JSON никак иначе
        return "You are an assistant that always provides responses in JSON format.\n" + prompt + "\nEnsure the output is valid JSON and matches the specified schema, if provided."

    def create_prompt(self, response_list, codes_list):
        formatted_prompt = self.sys_prompt.format(
            subject=self.prompts['subject'],
            first_question=self.prompts['first_question'],
            second_question=self.prompts['second_question'],
            # question=self.prompts['question'],
            # response_gist=self.prompts['response_gist'],
            not_valid_code=self.prompts['not_valid_code'],
            additional_requirements=self.prompts['not_valid_code'],
            codes_list=codes_list,
            response_list=response_list,
        )
        formatted_prompt = self.json_prompt(formatted_prompt)
        return formatted_prompt

    def call_llm(self, formatted_prompt: str, temperature: float):
        chat_completion = self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            temperature=temperature,
            model=self.model_name,
        )
        return chat_completion

    def classifier_responses(self, codes_list, response_list, temperature: float = 0.001):
        response_list = [f'{i+1}. {response}'for i,
                         response in enumerate(response_list)]
        formatted_prompt = self.create_prompt(response_list, codes_list)
        try_count = 3
        is_good_result = False
        while try_count > 0 and is_good_result == False:
            chat_completion = self.call_llm(
                formatted_prompt=formatted_prompt, temperature=temperature)
            try_count -= 1
            try:
                assistant_response = chat_completion.choices[0].message.content
                json_result = json.loads(assistant_response)
                is_good_result = True
            except json.JSONDecodeError as e:
                try:
                    assistant_response = chat_completion.choices[0].message.content
                    assistant_response = assistant_response.replace('```', '').replace('JSON', '').replace('json',
                                                                                                           '').strip()
                    json_result = json.loads(assistant_response)
                    is_good_result = True
                except json.JSONDecodeError as e:
                    try:
                        assistant_response = chat_completion.choices[0].message.content
                        assistant_response = assistant_response.replace('```', '').replace('JSON', '').replace('json',
                                                                                                               '').replace(
                            '"', '""').strip()
                        json_result = json.loads(assistant_response)
                        is_good_result = True
                    except json.JSONDecodeError as e:
                        print("Error: Invalid JSON response")
                        print(assistant_response)
        if not is_good_result:
            print('Не получилось обработать часть данных')
            return None
        try:
            response_codes_df = self.json_to_pandas(json_result)
        except:
            print('Не удалось перевести данные в датафрейм')
            print(f'json_result: {json_result}')
            return pd.DataFrame()
        return response_codes_df

    def get_OpenAI_client(self):
        if self.llm_type == 'OpenAI':
            openai.api_key = self.tokkens_config[self.llm_type]['OPENAI_API_KEY']
            openai_client = OpenAI(
                api_key=self.tokkens_config[self.llm_type]['OPENAI_API_KEY'],
                # http_client=httpx.Client(proxy=proxy_url)
            )
        else:
            raise 'Unknow type LLM'
        return openai_client


def log_time(message: str, last_time):
    current_time = time.time()
    print(f'{message}: {round(current_time - last_time)} сек.')
    return current_time


if __name__ == '__main__':
    start_time = last_time = time.time()
    PATH = ""

    # Токены для LLM
    file_tokens = PATH + "llm_tokens.json"
    with open(file_tokens, 'r', encoding='utf-8') as f:
        tokkens_config = json.load(f)

    # load_dotenv()
    # proxy_url = os.environ.get("OPENAI_PROXY_URL")
    # client = OpenAI() if proxy_url is None or proxy_url == "" else OpenAI(http_client=httpx.Client(proxy=proxy_url))

    # Промпты
    file_prompts = PATH + "sys_prompts.json"
    with open(file_prompts, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    last_time = log_time('Загрузили конфиг: ', last_time)

    answers_df = pd.read_csv(PATH + 'responses.csv', sep=';')

    is_with_codekey = askyesno(
        title="Наличие кодового ключа", message="Есть ли кодовый ключ?")

    if is_with_codekey:
        codes_df = pd.read_csv(PATH + 'codes.csv', sep=';')
        last_time = log_time(f'Загрузили данные ответы: {len(answers_df)} шт., классификатор: {len(codes_df)} шт.',
                             last_time)
    else:
        last_time = log_time(f'Загрузили данные ответы: {len(answers_df)} шт.',
                             last_time)

    # Инициализируем класс предсказания
    response_classifier = ResponseClassifier(
        tokkens_config=tokkens_config,
        prompts=prompts,
        with_codekey=is_with_codekey,
        # model_name='gpt-4o-mini-2024-07-18',
        model_name='gpt-4o',
        llm_type='OpenAI'
    )

    start_pos = 0
    batch_size = 30

    codes_list = ""
    if is_with_codekey:
        codes_list = "Код;Категория\n" + '\n'.join(
            codes_df[['Код', 'Категория']].apply(lambda x: ';'.join(list(map(str, x))), axis=1).values)

    result_df = pd.DataFrame()
    for i in trange(0, len(answers_df), batch_size):
        current_batch_df = answers_df[start_pos:(start_pos + batch_size)]
        response_list = current_batch_df['Ответы'].replace(
            r'\n', ' ', regex=True).replace(r'\r', ' ', regex=True).values
        response_codes_df = response_classifier.classifier_responses(codes_list=codes_list, response_list=response_list,
                                                                     temperature=0.0)
        response_codes_df['Ответы'] = current_batch_df['Ответы'].values
        result_df = pd.concat([result_df, response_codes_df])
        start_pos += batch_size
    result_df = result_df.reset_index().drop(columns='index')
    # result_df[['Ответы', 'code_1', 'code_2', 'code_3', 'code_4', 'code_5']].to_csv(
    #     f'result_v7_8_5.csv', index=False)

    result_df[['Ответы', 'code_1', 'code_2', 'code_3',
               'code_4', 'code_5', 'code_6', 'code_7',
               'code_8', 'code_9', 'code_10']].to_excel('result.xlsx')
