# 참고: minimind-kr에서는 tokenizer 재훈련이 사실상 필수입니다. 원본 minimind의 6,400 vocab BPE는
# 중국어에 최적화되어 한국어 커버리지가 매우 낮습니다. 아래 스크립트로 한국어 코퍼스를 학습해 새 tokenizer를 만드세요.
#
# 추천 한국어 코퍼스:
#   - 위키피디아 한국어 덤프 (https://dumps.wikimedia.org/kowiki/)
#   - 나무위키 덤프
#   - AI Hub 한국어 대화/뉴스 데이터셋 (https://aihub.or.kr/)
#   - KoAlpaca (https://github.com/Beomi/KoAlpaca)
#   - KULLM (https://github.com/nlpai-lab/KULLM)
#   - KLUE / KorQuAD
#
# 사용 예:
#   python train_tokenizer.py --data_path ../dataset/ko_corpus.jsonl --tokenizer_dir ../model --vocab_size 8000
#
# 주의: tokenizer를 바꾸면 pretrain부터 모든 학습을 처음부터 다시 돌려야 합니다.
# 상위 프로젝트의 사전 학습 weight는 vocab이 달라 재사용할 수 없습니다.

import os
import json
import argparse
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 기본값 (CLI로 재정의 가능)
DEFAULT_DATA_PATH = '../dataset/ko_corpus.jsonl'
DEFAULT_TOKENIZER_DIR = '../model_learn_tokenizer/'
DEFAULT_VOCAB_SIZE = 8000  # 한국어 커버리지를 위해 원본(6400)보다 상향
DEFAULT_SAMPLE_LIMIT = 100000  # 학습에 사용할 최대 라인 수 (0이면 전체)
SPECIAL_TOKENS_NUM = 36

def get_texts(data_path, sample_limit=0):
    """JSONL 파일에서 텍스트를 읽어냅니다.
    - SFT 형식({"conversations": [{"role":..., "content":...}]})의 경우 content를 모아 반환
    - Pretrain 형식({"text": "..."})의 경우 text 필드를 반환
    """
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if sample_limit and i >= sample_limit:
                break
            try:
                data = json.loads(line)
                if 'conversations' in data:
                    contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                    if contents:
                        yield "\n".join(contents)
                elif 'text' in data and data['text']:
                    yield str(data['text'])
            except json.JSONDecodeError:
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, sample_limit=DEFAULT_SAMPLE_LIMIT, special_tokens_num=SPECIAL_TOKENS_NUM):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]

    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]  # 향후 확장을 위한 예약 토큰
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens
    )
    texts = get_texts(data_path, sample_limit=sample_limit)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }

    # chat_template은 한국어 tool-calling 시스템 메시지로 지역화된 버전을 사용합니다.
    # 기존 model/tokenizer_config.json의 chat_template를 그대로 재사용하도록 유지하되,
    # 새 tokenizer 디렉터리에도 동일하게 기록합니다.
    chat_template = (
        "{%- if tools %}\n"
        "    {{- '<|im_start|>system\\n' }}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- messages[0].content + '\\n\\n' }}\n"
        "    {%- endif %}\n"
        "    {{- \"# 도구\\n\\n사용자의 질의를 돕기 위해 하나 이상의 함수를 호출할 수 있습니다.\\n\\n<tools></tools> XML 태그 안에 함수 시그니처가 제공됩니다:\\n<tools>\" }}\n"
        "    {%- for tool in tools %}\n"
        "        {{- \"\\n\" }}\n"
        "        {{- tool | tojson }}\n"
        "    {%- endfor %}\n"
        "    {{- \"\\n</tools>\\n\\n각 함수 호출에 대해 함수 이름과 인자를 담은 JSON 객체를 <tool_call></tool_call> XML 태그 안에 반환하세요:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
        "{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
        "{%- for message in messages[::-1] %}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
        "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
        "        {%- set ns.multi_step_tool = false %}\n"
        "        {%- set ns.last_query_index = index %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.content is string %}\n"
        "        {%- set content = message.content %}\n"
        "    {%- else %}\n"
        "        {%- set content = '' %}\n"
        "    {%- endif %}\n"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {%- set reasoning_content = '' %}\n"
        "        {%- if message.reasoning_content is string %}\n"
        "            {%- set reasoning_content = message.reasoning_content %}\n"
        "        {%- else %}\n"
        "            {%- if '</think>' in content %}\n"
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
        "            {%- endif %}\n"
        "        {%- endif %}\n"
        "        {%- if true %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
        "        {%- endif %}\n"
        "        {%- if message.tool_calls %}\n"
        "            {%- for tool_call in message.tool_calls %}\n"
        "                {%- if (loop.first and content) or (not loop.first) %}\n"
        "                    {{- '\\n' }}\n"
        "                {%- endif %}\n"
        "                {%- if tool_call.function %}\n"
        "                    {%- set tool_call = tool_call.function %}\n"
        "                {%- endif %}\n"
        "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
        "                {{- tool_call.name }}\n"
        "                {{- '\", \"arguments\": ' }}\n"
        "                {%- if tool_call.arguments is string %}\n"
        "                    {{- tool_call.arguments }}\n"
        "                {%- else %}\n"
        "                    {{- tool_call.arguments | tojson }}\n"
        "                {%- endif %}\n"
        "                {{- '}\\n</tool_call>' }}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        "    {%- elif message.role == \"tool\" %}\n"
        "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
        "            {{- '<|im_start|>user' }}\n"
        "        {%- endif %}\n"
        "        {{- '\\n<tool_response>\\n' }}\n"
        "        {{- content }}\n"
        "        {{- '\\n</tool_response>' }}\n"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- endif %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if open_thinking is defined and open_thinking is true %}\n"
        "        {{- '<think>\\n' }}\n"
        "    {%- else %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}"
    )

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": chat_template,
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer 학습이 완료되었습니다.")

def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "당신은 훌륭한 챗봇이며, 항상 정확한 답변을 제공합니다!"},
        {"role": "user", "content": '당신은 어디에서 왔나요?'},
        {"role": "assistant", "content": '저는 달에서 왔습니다.'},
        {"role": "user", "content": '정말 어디에서 왔나요?'},
        {"role": "assistant", "content": '저는 지구에서 왔습니다.'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-' * 100)
    print(new_prompt)
    print('-' * 100)
    print('tokenizer 사전 길이:', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder 길이:', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder 일관성:', response == new_prompt, "\n")
    print('-' * 100)
    print('압축률 테스트 (Chars/Tokens):')
    test_texts = [
        # 한국어 샘플 (약 200자)
        "인공지능은 컴퓨터 과학의 한 분야로, 지능의 본질을 이해하고 인간의 지능과 유사한 방식으로 반응할 수 있는 새로운 지능형 기계를 만들어내려는 시도입니다. 이 분야의 연구에는 로봇공학, 음성 인식, 이미지 인식, 자연어 처리, 전문가 시스템 등이 포함됩니다. 인공지능은 탄생 이래로 이론과 기술이 점점 성숙해지고 있으며, 응용 분야도 계속 확장되고 있어 앞으로 인공지능이 가져올 기술 제품이 인간 지혜의 그릇이 될 것이라 상상해 볼 수 있습니다.",
        "우주 항해는 항성계 내부, 또는 항성계 간의 우주 공간에서 이루어지는 항해를 말합니다. 우주 공간은 극히 광활하기 때문에 전통적인 화학 로켓 추진으로는 항성 간 항해에 역부족입니다. 과학자들은 이온 추진기, 핵열 로켓, 심지어 반물질을 에너지원으로 사용하는 방안 등 여러 방법을 제안했습니다. 또한 곡률 추진이나 웜홀 여행 같은 SF적 개념도 이론 물리학 연구에서 반복적으로 논의되고 있습니다.",
        # 영어 샘플 (약 200 단어)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply.",
        # 혼합 샘플
        "Python은 간결한 문법과 강력한 생태계로 유명한 고급 프로그래밍 언어입니다. It is widely used in data science, machine learning, and web development. 개발자는 NumPy, Pandas, PyTorch 같은 라이브러리를 활용해 복잡한 애플리케이션을 빠르게 구축할 수 있습니다. Whether you are a beginner or an expert, Python offers something for everyone.",
    ]

    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"샘플 {i+1} | 문자 수: {char_count:4} | Tokens: {token_count:3} | 압축률: {compression_ratio:.2f}")

    print(f"평균 압축률: {total_compression / len(test_texts):.2f}")
    print('-' * 100)
    print('스트리밍 디코딩(바이트 버퍼) 테스트:')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '�' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="minimind-kr 한국어 tokenizer 학습")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help="한국어 코퍼스 JSONL 경로 ({\"text\": ...} 또는 {\"conversations\": [...]} 형식)")
    parser.add_argument('--tokenizer_dir', type=str, default=DEFAULT_TOKENIZER_DIR,
                        help="새 tokenizer를 저장할 디렉터리 (예: ../model)")
    parser.add_argument('--vocab_size', type=int, default=DEFAULT_VOCAB_SIZE,
                        help="어휘 크기. 한국어는 원본(6400)보다 8000~16000 정도가 권장됩니다.")
    parser.add_argument('--sample_limit', type=int, default=DEFAULT_SAMPLE_LIMIT,
                        help="학습에 사용할 최대 라인 수 (0이면 전체 파일 사용)")
    parser.add_argument('--skip_eval', action='store_true', help="학습 후 eval_tokenizer 단계를 건너뜁니다.")
    args = parser.parse_args()

    train_tokenizer(args.data_path, args.tokenizer_dir, args.vocab_size, sample_limit=args.sample_limit)
    if not args.skip_eval:
        eval_tokenizer(args.tokenizer_dir)
