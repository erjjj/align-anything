from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from torch.utils.data.distributed import DistributedSampler
from align_anything.configs.template import ChatTemplate
from align_anything.models.pretrained_model import load_pretrained_models
import json
from tqdm import tqdm
device = "cuda" # the device to load the model onto
def get_response_from_model(model,tokenizer,prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # generated_ids:Tensor其中包含输入数据，可以直接被用于reward_model打分，这一部分通过下面验证，
    # 可以发现用于训练reward_model的input_ids数据和generated_ids是一个结构的
    # 因为它们decode后都是包含前置system、输入user、回答response三个部分
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # 把回答全部输出出来，更美观
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_ids,response

def get_rewards_from_two_model(model1,tokenizer1,model2,tokenizer2,prompt,model_reward):
    generated_ids_1,response_1=get_response_from_model(model1,tokenizer1,prompt)
    generated_ids_2,response_2=get_response_from_model(model2,tokenizer2,prompt)
    output_1=model_reward(**{'input_ids':generated_ids_1})
    output_2=model_reward(**{'input_ids':generated_ids_2})
    return response_1,response_2,output_1.end_scores,output_2.end_scores

# 原始模型提取
model_raw = AutoModelForCausalLM.from_pretrained(
    "Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
print("Raw model loaded successfully.")

tokenizer_raw = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")
print("Raw tokenizer loaded successfully.")

# dpo强化学习训练后的模型提取
model_dpo = AutoModelForCausalLM.from_pretrained(
    "../outputs/qwen_2_5_dpo_727/slice_end",
    torch_dtype="auto",
    device_map="auto"
)
print("DPO model loaded successfully.")
tokenizer_dpo = AutoTokenizer.from_pretrained("../outputs/qwen_2_5_dpo_727/slice_end")
print("DPO tokenizer loaded successfully.")
# 加载奖励模型model_reward
model_reward, tokenizer_reward, processor_reward = load_pretrained_models(
    '../outputs/qwen_2_5_rm/slice_15208',
    model_max_length=512,
    padding_side='right',
    trust_remote_code=True,
    is_reward_model=True,
    processor_kwargs=None,
    auto_device_mapping=True
)
print("Reward model loaded successfully.")
import pandas as pd

df = pd.read_json("../../assets/text_to_text/hw/val.json", lines=True)  
data_batch=[]
for idx, question in enumerate(df['question']):
    data_batch.append(question)
    
print(f"Loaded {len(data_batch)} prompts from the dataset.")
# 下面应该根据特定的eval数据集批量生成回复并计算reward
def get_batch_rewards_from_two_model(model1,tokenizer1,model2,tokenizer2,data_batch,model_reward,output_file='version1.json'):
    results=[]
    for prompt in tqdm(data_batch, desc="Evaluating prompts"):
        response_raw,response_dpo,reward_raw,reward_dpo=\
            get_rewards_from_two_model(model1,tokenizer1,model2,tokenizer2,prompt,model_reward)
        result = {
            "prompt": prompt,
            "response_raw": response_raw,
            "response_dpo": response_dpo,
            "reward_raw": reward_raw[0].item(),
            "reward_dpo": reward_dpo[0].item()
        }

        results.append(result)

    # 写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"已保存结果到 {output_file}")
    return results

res=get_batch_rewards_from_two_model(
    model_raw,tokenizer_raw,
    model_dpo,tokenizer_dpo,
    data_batch,
    model_reward,
    output_file='version1.json'
)
