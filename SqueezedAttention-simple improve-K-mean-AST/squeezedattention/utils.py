import os

# This is the customized building prompt for chat models
def build_chat(prompt, model_name, noquery=False, tokenizer=None):
    """
    Ap chat template.

    CANH BAO DA GAP (23/8): bang cung ben duoi CHI biet hai ho model — "LWM" va "longchat".
    Voi moi model khac no TRA VE PROMPT NGUYEN XI, khong bao gi. Hau qua: lan do
    `--force_chat` dau tien cho ra dung 17,6 — bang het lan khong template — vi template
    chua bao gio duoc ap. Muon biet template co that su duoc ap khong thi so do dai chuoi
    truoc/sau, dung tin vao ten co.

    Nen uu tien `tokenizer.apply_chat_template` khi co tokenizer: no dung dung template ma
    chinh model duoc huan luyen (ChatML voi Qwen), thay vi doan theo ten.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=not noquery)

    if noquery:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt}"
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            prompt = conv.get_prompt()

    else:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt} ASSISTANT: "
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
    return prompt

def truncate_fn(prompt, prompt_noquery, tokenizer, max_length, dataset, device,
                model_name=None, force_chat=False):
    # model_name/force_chat la THEM VAO, mac dinh giu nguyen hanh vi cu:
    #   - moi loi goi cu khong truyen gi -> force_chat=False
    #   - lcc/repobench-p nam trong danh sach loai tru -> nhanh chat khong chay
    # Chung sinh ra de (a) sua bug `model_name` khong ton tai trong nhanh chat,
    # va (b) cho phep BAT chat template co chu dich khi do model Instruct.
    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    tokenized_prompt_noquery = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids[0]

    # truncate based on length of prompt with query
    len_tokenized_prompt = len(tokenized_prompt)
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)

        # compute num tokens removed and subtract from sp_len
        tokens_removed = len(tokenized_prompt) - 2*half

        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    else:
        tokens_removed = 0

    # incorporate chat template for shared prefix length
    if force_chat or dataset not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
        # BUG CU: `model_name` khong phai tham so cua ham va khong co trong scope ->
        # nhanh nay ne'm NameError. Chua ai vap vi Phase 0/1 chi dung lcc/repobench-p,
        # ca hai deu nam trong danh sach loai tru.
        if model_name is None:
            raise ValueError(
                "truncate_fn: can `model_name` khi ap chat template "
                f"(dataset={dataset}, force_chat={force_chat})")
        before = len(prompt)
        prompt = build_chat(prompt, model_name, tokenizer=tokenizer)
        prompt_noquery = build_chat(prompt_noquery, model_name, noquery=True,
                                    tokenizer=tokenizer)
        if force_chat and len(prompt) == before:
            raise ValueError(
                f"--force_chat duoc bat nhung template KHONG duoc ap cho '{model_name}': "
                f"tokenizer khong co chat_template va ten model khong khop bang cung. "
                f"Chay tiep se do lai dung cau hinh cu ma tuong la da doi.")

    # compute shared prefix length
    input_ids_prompt_only = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids.to(device)
    shared_prefix_length = input_ids_prompt_only.shape[1]

    return prompt, shared_prefix_length - tokens_removed
