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
    tmp_file = 'tmp_' + subs_out
    operation = ['ffmpeg', '-i', video_in, '-map', f'0:s:{subs_track}', tmp_file]
    print(operation)
    status = subprocess.run(operation)
    merge_subtitles(tmp_file, subs_out)
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

#chatGPT
def translate_use_chatGPT(subtitles, language='Chinese') -> List[str]:
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
    #get api_key from env

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    count = 2
    while count > 0:

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        text = response.choices[0].message.content


        try:
            t = re.findall(r'```json\n?(.+)\n?```', text, re.DOTALL)
            if t[0]:
                try:
                    k = json.loads(t[0])
                except:
                    count -= 1
                    continue
                if len(k) != h+1:
                    print("翻译行数不对，请重新翻译")
                    count -= 1
                return k
            else:
                count -= 1
        except:
            try:
                t = json.loads(text)
                if len(t) != h+1:
                    print("翻译行数不对，请重新翻译")
                    count -= 1
                return t
            except:
                count -= 1
    return []


if __name__ == '__main__':
    mkv_file = '/Users/xucanjie/Downloads/Nicky.Ricky.Dicky.And.Dawn.S01E02.1080p.NF.WEB-DL.DDP2.0.x264-LAZY.mkv'
    input_file = 'input_subtitle_file.ass'  # Update this to your actual input file path
    output_file = 'merged_subtitle_file.ass'  # Update this to your desired output file path
    #extract_from_video_mkv(mkv_file, 0, input_file)
    #print(read_subtitles(input_file))
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Dialogue: '):
                continue
            lines.append(line)
    for i in get_batches(read_subtitles(input_file)):
        e = i['current_batch']
        rs = translate_use_chatGPT(i)

        if len(rs) > 0 and len(rs) == len(e):
            for index,k in enumerate(rs):
                print(index, k)
                try:
                    l = e[index][0]+','+ e[index][1] + afterstyle(k) + '\n'
                    #l = e[index][0]+','+ k + '\n'
                    lines.append(l)
                except:
                    print(index,k, e)
                print(l)
        else:
            for ori in e:
                l = ori[0] + ',' + ori[1] + '\n'
                lines.append(l)
    print(lines)
    with open(output_file, 'w',encoding='utf-8') as f:
        f.writelines(lines)
