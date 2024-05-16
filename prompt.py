import lang
def gen_prompt_en(num_qs, header, content):
    return f"""
            [INST]
            Generate {num_qs} multiple-choice questions from the given paragraph.
            Respond with the following template:
            
            <QUESTION>
            <CHOICE A>
            <CHOICE B>
            <CHOICE C>
            <CHOICE D>
            <CORRECT CHOICE LETTER>
            "---"

            Do NOT give any explanations.\n
            {header}
            {content}\n

            [/INST]
        """

def gen_prompt_vi(num_qs, header, content):
    return f"""
            [INST]
            Tạo các câu hỏi trắc nghiệm 4 lựa chọn từ đoạn văn sau.
            {content}\n
            [/INST]
        """