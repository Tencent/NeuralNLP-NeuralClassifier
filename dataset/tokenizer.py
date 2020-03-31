import jieba
import jieba.posseg as pseg
from pytorch_pretrained_bert import BertTokenizer

class Tokenizer:
    def tokenize(self, text:str) -> dict:
        #out = {'doc_token': List[str]}
        raise NotImplementedError()


class TokenizerJieba(Tokenizer):
    def __init__(self, use_paddle:bool=False):
        self.use_paddle = use_paddle
        if use_paddle:
            jieba.enable_paddle()

    def tokenize(self, text:str) -> dict:
        output = {}
        output['doc_token'] = list(jieba.cut(text, use_paddle=self.use_paddle))
        return output


class TokenizerJiebaPos():
    def __init__(self, use_paddle=False):
        self.enable_paddle = use_paddle
        if use_paddle:
            jieba.enable_paddle()

    def tokenize(self, text:str) -> dict:
        output = {}
        output['doc_token'], output['doc_pos'] = [list(e) for e in zip(*pseg.cut(text, use_paddle=self.enable_paddle))]
        return output


class TokenizerBERT(Tokenizer):
    def __init__(self, language:str='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(language)

    def tokenize(self, text:str) -> dict:
        output = {}
        output['doc_token'] = ['[CLS]'] + self.tokenizer.tokenize(text)[:512] + ['[SEP]']
        return output


def get_tokenizer(name:str='jieba', **kwg):
    init_func = {
            'jieba': TokenizerJieba,
            'jieba_pos': TokenizerJiebaPos,
            'bert': TokenizerBERT
            }
    return init_func[name](**kwg)

if __name__ == '__main__':
    text = "目前已有崩坏3、QQ飞车手游、节奏大师等近百款游戏已支持144Hz模式"
    tokenizer = get_tokenizer('bert')
    print(tokenizer.tokenize(text))

    import ipdb; ipdb.set_trace()

