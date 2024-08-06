import json
import re
import subprocess
from typing import List
from openai import OpenAI
import os


def merge_subtitles(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    last_entry = None

    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.strip().split(',', 9)  # Split only on first 9 commas
            if len(parts) == 10:
                timing = parts[1] + ',' + parts[2]  # Combine start and end time for comparison
                if last_entry and last_entry['timing'] == timing:
                    # Append text to the last entry if timings match
                    last_entry['text'] += ' ' + parts[-1].strip()
                    continue
                else:
                    # Otherwise, start a new entry
                    if last_entry:
                        # Format and add the completed entry to results
                        results.append(','.join(last_entry['parts'][:-1] + [last_entry['text']]) + '\n')
                    last_entry = {'timing': timing, 'parts': parts, 'text': parts[-1].strip()}
            else:
                # If line is malformed, add it directly to results
                results.append(line)
        else:
            # Directly add non-dialogue lines to results
            results.append(line)

    # Ensure the last entry is added
    if last_entry:
        results.append(','.join(last_entry['parts'][:-1] + [last_entry['text']]) + '\n')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(results)


def extract_from_video_mkv(video_in: str, subs_track: int, subs_out: str) -> bool:
    base_name = os.path.basename(subs_out)
    print(base_name)
    tmp_file = f'tmp_{base_name}.ass'
    operation = ['ffmpeg', '-i', video_in, '-map', f'0:s:{subs_track}', tmp_file]
    print(operation)
    status = subprocess.run(operation)
    merge_subtitles(tmp_file, subs_out)
    os.remove(tmp_file)
    return status.returncode == 0



def read_subtitles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    subtitles = []
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.strip().split(',', 9)
            if len(parts) == 10:
                subtitles.append([','.join(parts[:-1]), parts[-1].strip()])  # 提取对话文本部分
    return subtitles


def get_batches(subtitles, batch_size=10):
    for i in range(0, len(subtitles), batch_size):
        yield {
            'previous_context': subtitles[max(i-5, 0):i],
            'current_batch': subtitles[i:i + batch_size],
            'next_context': subtitles[i + batch_size:i + batch_size + 2]
        }


open_style = ''
close_style = ''

def afterstyle(text: str, scale: int = 80, alpha: int = 50) -> str:
    alpha_scale = round(alpha * 2.55)
    alpha_hex = f'{alpha_scale:x}'
    open_style = f'\\N{{\\fscx{scale}\\fscy{scale}\\alpha&H{alpha_hex}&}}'
    close_style = '\\N{\\fscx100\\fscy100\\alpha&H00&}'
    return f'{open_style}{text}{close_style}'


def remove_numbers(s):
    # 使用正则表达式匹配以数字和冒号开头的部分
    return re.sub(r'^\d+:\s*', '', s, flags=re.MULTILINE)


def chat_with_deepseek(system_prompt, user_prompt) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    return text

def chat_with_qwen2(system_prompt, user_prompt) -> str:
    api_key = os.getenv("QWEN_OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    return text

#qwen2
def translate_use_llm(subtitles, language='Chinese', count=2, llm='deepseek') -> List[str]:
    system_prompt = f'你是一个专业的字幕翻译员，你的任务是将以下英文字幕翻译成{language}, 你需要根据上下文进行翻译。'
    pre_text = ''
    cur_text = ''
    nex_text = ''
    h = 0
    for i in subtitles['previous_context']:
        pre_text += f'{i[1]}\n'
    for m,c_text in enumerate(subtitles['current_batch']):
        cur_text += f'{m+1}: {c_text[1]}\n'
        h = m
    for i in subtitles['next_context']:
        nex_text += f'{i[1]}\n'
    prompt = (f"上文是：\n{pre_text} \n当前文本:\n{cur_text} \n下文是：\n{nex_text} \n"
              f"你需要将当前文本逐行按顺序翻译成{language},总共有{h+1}行，不能自行合并行，有重复的行也不能忽略， 并去掉一些特殊符合。如：-，{{\i0}}, {{\i1}}等\n每个当前文本对应一个翻译文本。\n"
              f"只返回json格式[译文1，译文2...]，译文不要显示行号，不要包含其他内容。")
    print(prompt)

    while count > 0:
        text = ''
        if llm == 'qwen2':
            text = chat_with_qwen2(system_prompt, prompt)
        if llm == 'deepseek':
            text = chat_with_deepseek(system_prompt, prompt)
        print(text)
        try:
            pattern = r"^.*?(\[.*\]).*$"
            matches = re.findall(pattern, text, re.DOTALL)
            print(matches)
            if matches:
                try:
                    k = json.loads(matches[0])
                except:
                    count -= 1
                    continue
                if len(k) != h + 1:
                    print("翻译行数不对，请重新翻译")
                    count -= 1
                return [remove_numbers(s) for s in k]
            else:
                count -= 1
        except:
            count -= 1
        count -= 1
        return []



def generate_subtitle_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mkv'):
                base_name = os.path.splitext(file)[0]
                mkv_file_path = os.path.join(root, file)
                subtitle_file_path = os.path.join(root, f'{base_name}.ass')
                tmpsubfile = os.path.join(root, f'tmp_{base_name}.ass')
                extract_from_video_mkv(mkv_file_path,0, tmpsubfile)
                lines = []
                with open(tmpsubfile, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith('Dialogue: '):
                                continue
                            lines.append(line)
                for i in get_batches(read_subtitles(tmpsubfile)):
                    e = i['current_batch']
                    rs = translate_use_llm(i)
                    if len(rs) > 0 and len(rs) == len(e):
                        for index, k in enumerate(rs):
                            print(index, k)
                            try:
                                l = e[index][0] + ',' + e[index][1] + afterstyle(k) + '\n'
                                lines.append(l)
                            except:
                                print(index, k, e)
                            print(l)
                    else:
                        print("in qwen2---------------")
                        rs = translate_use_llm(i, llm='qwen2')
                        if len(rs) > 0 and len(rs) == len(e):
                            for index, k in enumerate(rs):
                                print(index, k)
                                try:
                                    l = e[index][0] + ',' + e[index][1] + afterstyle(k) + '\n'
                                    lines.append(l)
                                except:
                                    print(index, k, e)
                                print(l)
                        else:
                            for ori in e:
                                l = ori[0] + ',' + ori[1] + '\n'
                                lines.append(l)

                print(lines)
                with open(subtitle_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                os.remove(tmpsubfile)


if __name__ == '__main__':
    import sys
    generate_subtitle_file(".")
