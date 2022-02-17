# -*- coding: utf-8 -*-
from pyvi.pyvi import ViTokenizer # pyvi == 0.0.8.0
from g2p_en import G2p # g2p-en == 2.0.0


fundamental = 'aăâeêioôơuưy' \
              'áắấéếíóốớúứý' \
              'àằầèềìòồờùừỳ' \
              'ảẳẩẻểỉỏổởủửỷ' \
              'ãẵẫẽễĩõỗỡũữỹ' \
              'ạặậẹệịọộợụựỵ'
phone = {'y': 'iy', 'i': 'ii', 'e': 'eh', 'u': 'uw', 'ă': 'az', 'ph': 'f', 'gi': 'y', 'th': 'tt', 'r': 'rz', 'ch': 'ck',
         'd': 'y', 'h': 'hh', 'ngh': 'ng', 's': 'sr', 'g': 'gh', 'qu': 'w', 'đ': 'd', 'c': 'k', 'x': 's', }
g2p = G2p()
rhyme = {}
for idx, i in enumerate(list(fundamental)):
    rhyme[i] = idx

def word_to_phoneme(word):
    phonemes = []
    token = []

    # get tone
    for i in word:
        if i in fundamental[12:]:
            word = word.replace(i, fundamental[rhyme[i]%12])
            token.append(word)
            token.append(int(rhyme[i]/12))
            break

    if len(token) == 0:
        token.append(word)
        token.append(0)
    length = len(token[0])
    # get phoneme
    if token[0] == "gi" and (token[1] == 2 or token[1] == 3):
        phonemes = ['y', 'iy', token[1]]
    elif token[0] == "gin" and token[1] == 2:
        phonemes = ['y', 'iy', 'n', token[1]]
    else:
        for id, char in enumerate(token[0]):

            if char == 'a': #and (id == 0 or (id > 0 and token[0][id - 1] not in ['y', 'i'])):
                if id == length - 1 and (token[0][id - 1] not in fundamental or length == 1):
                    phonemes.append('at')
                elif id + 1 < length  and token[0][id + 1] in fundamental:
                    phonemes.append('az')
                else:
                    phonemes.append('ax')
            elif char == 'o':
                if (id >= 1 and (token[0][id - 1] + char) == 'oo'):
                    phonemes.pop()
                    if (token[1] == 0 or token[1] == 1 or token[1] == 2):
                        phonemes.append('ao' + str(token[1]))
                    else:
                        phonemes.append('ao')
                elif id == len(token[0]) - 1:
                    if (token[1] == 0 or token[1] == 1 or token[1] == 2):
                        phonemes.append('ao' + str(token[1]))
                    else:
                        phonemes.append('ao')
                else:
                    phonemes.append('oz')
            elif (char == 'y'  or char == 'e' or (char == 'u' and id == 0) or (char == 'u' and id > 0 and token[0][id-1]!='q')) and (token[1] == 0 or token[1] == 1 or token[1] == 2):
                phonemes.append(phone[char] + str(token[1]))
            elif id > 0:
                #print(token[0][id-1] + char)
                if (phonemes[-1]+char == 'ngh'):
                    phonemes.pop()
                    phonemes.append('ng')
                elif (token[0][id-1] + char) in ['ng', 'gh', 'nh', 'ch', 'tr', 'th', 'ph', 'kh', 'gi', 'qu']:
                    x = token[0][id-1]
                    phonemes.pop()
                    if (x + char) in phone:
                        phonemes.append(phone[x+char])
                    else:
                        phonemes.append(x+char)
                elif ((token[0][id-1] + char) in  ['ya', 'yê']):
                    phonemes.pop()
                    phonemes.append('ie')
                elif (token[0][id-1] + char) in ['ia','iê']:
                    #print(token[0][id-1] + char)
                    if phonemes[-1] != 'y':
                        phonemes.pop()
                        phonemes.append('ie')
                    else:
                        phonemes.append('ie')
                elif ((token[0][id-1] + char) in ['uô', 'ua']):
                    if phonemes[-1]=='w':
                        if char in phone:
                            phonemes.append(phone[char])
                        else:
                            phonemes.append(char)
                    else:
                        phonemes.pop()
                        phonemes.append('uo')

                elif ((token[0][id-1] + char) in ['ươ', 'ưa']):
                    phonemes.pop()
                    phonemes.append('wx')
                # elif char == 'i' and len(phonemes) >= 2 and id == length - 1:
                #     phonemes.append('ii')
                else:
                    if char in phone:
                        phonemes.append(phone[char])
                    else:
                        phonemes.append(char)
            elif char in phone:
                phonemes.append(phone[char])
            else:
                phonemes.append(char)
            #print(token[0][id], phonemes[-1])
        phonemes.append(token[1])
    if phonemes[-1] != 0:
        syllable = '-'.join([(i + str(phonemes[-1])) for i in phonemes[:-1]])
    else:
        syllable = '-'.join(phonemes[:-1])

    return syllable

en_dict = {}
with open('text/cmudict_vn', 'r') as f:
    for line in f:
        word = line.split('  ')
        en_dict[word[0].strip().lower()] = '-'.join(word[1].lower().strip().split(' '))
vi_dict = {}
with open('text/vocab_phoneme.txt', 'r') as f:
    for line in f.readlines():
        vi_dict[line.strip()] = 1

name_dict = {}
with open('text/name.txt', 'r') as f:
    for line in f.readlines():
        name_dict[line.lower().strip()] = 1
def pronounciation(sent):
    word_list = []
    w = ''
    for id, c in enumerate(sent):
        if w == '':
            w += c
        elif id > 0:
            if (sent[id - 1] + c) in ['ch', 'tr', 'ng', 'kh', 'nh', 'ph', 'th', 'gi', 'gh']:
                w += c
                if id == len(sent) - 1:
                    word_list.append(w)
                    w = ''
            elif (sent[id - 1] not in fundamental and c in fundamental) \
                    or (sent[id - 1] in fundamental and c in fundamental):
                w += c
                if id == len(sent) - 1:
                    word_list.append(w)
                    w = ''
            elif (sent[id - 1] not in fundamental and c not in fundamental):
                if (sent[id - 1] + c) in ['ch', 'tr', 'ng', 'kh', 'nh', 'ph', 'th', 'gi', 'gh']:
                    w += c
                else:
                    word_list.append(w)
                    w = c
            elif (sent[id - 1] in fundamental and c not in fundamental):

                if id == len(sent) - 1:
                    word_list.append(w + c)
                    w = ''
                else:
                    if sent[id + 1] in fundamental:

                        word_list.append(w)
                        w = c
                    elif sent[id + 1] not in fundamental:
                        if c + sent[id + 1] in ['tr', 'kh', 'th', 'gi', 'gh', 'nh']:
                            word_list.append(w)
                            w = c
                        elif (c + sent[id + 1]) not in ['ch', 'ng', 'nh', 'ph']:
                            word_list.append(w + c)
                            w = ''

                        else:
                            if id + 2 < len(sent) and sent[id + 1] + sent[id + 2] == 'gi':
                                word_list.append(w + c)
                                w = ''
                            else:
                                if c + sent[id + 1] == 'ph':
                                    word_list.append(w)
                                    w = c
                                else:
                                    w += c
    if w != '':
        word_list.append(w)
    return word_list

unigram_v = {}
with open('text/unigram_vi.txt', 'r') as f:
    for line in f.readlines():
        unigram_v[line.split('\t')[0]] = int(line.split('\t')[1])

unigram_ambiguous_v = {}
with open('text/unigram_ambiguous_vi.txt', 'r') as f:
    for line in f.readlines():
        unigram_ambiguous_v[line.split('\t')[0]] = int(line.split('\t')[1])

unigram_e = {}
with open('text/unigram_en.txt', 'r') as f:
    for line in f.readlines():
        unigram_e[line.split('\t')[0]] = int(line.split('\t')[1])

bigram_v = {}
with open('text/bigram_ambiguous_vi.txt', 'r') as f:
    for line in f.readlines():
        bigram_v[line.split('\t')[0]] = int(line.split('\t')[1])

bigram_e = {}
with open('text/bigram_ambiguous_en.txt', 'r') as f:
    for line in f.readlines():
        bigram_e[line.split('\t')[0]] = int(line.split('\t')[1])

def lang_score(word, vi_score, en_score):
    #print(word, vi_score, en_score)
    if vi_score >= en_score:
        return word_to_phoneme(word)
    return en_dict[word]

def lang_process(idx, words):
    if words[idx] == 'a' and idx != len(words)-1:
        if (idx !=0 and len(words[idx-1]) == 1) or len(words[idx+1]) == 1:
            return en_dict[words[idx]]
        elif words[idx+1] in en_dict:
            return word_to_phoneme('ờ')
        else:
            return word_to_phoneme(words[idx])
    if idx == 0 and idx != len(words) - 1:
        vi_score = bigram_v[words[idx] + ' ' + words[idx + 1]]*1.0 / unigram_v[words[idx]] if words[idx] + ' ' + words[idx + 1] in bigram_v else 0
        en_score = bigram_e[words[idx] + ' ' + words[idx + 1]]*1.0 / unigram_e[words[idx]] if words[idx] + ' ' + words[idx + 1] in bigram_e else 0
        return lang_score(words[idx], vi_score, en_score)
    elif idx != 0 and idx == len(words) - 1:
        vi_score = bigram_v[words[idx - 1] + ' ' + words[idx]]*1.0 / unigram_v[words[idx-1]] if words[idx-1] + ' ' + words[idx] in bigram_v else 0
        en_score = bigram_e[words[idx - 1] + ' ' + words[idx]]*1.0 / unigram_e[words[idx-1]] if words[idx-1] + ' ' + words[idx] in bigram_e else 0
        return lang_score(words[idx], vi_score, en_score)
    elif idx == 0 and idx == len(words) - 1:
        return en_dict[words[idx]]
    else:
        vi_score = bigram_v[words[idx] + ' ' + words[idx + 1]]*1.0 / unigram_v[words[idx]] if words[idx] + ' ' + words[idx + 1] in bigram_v else 0
        vi_score += bigram_v[words[idx - 1] + ' ' + words[idx]]*1.0 / unigram_v[words[idx-1]] if words[idx-1] + ' ' + words[idx] in bigram_v else 0
        en_score = bigram_e[words[idx] + ' ' + words[idx + 1]]*1.0 / unigram_e[words[idx]] if words[idx] + ' ' + words[idx+1] in bigram_e else 0
        en_score += bigram_e[words[idx - 1] + ' ' + words[idx]]*1.0 / unigram_e[words[idx-1]] if words[idx-1] + ' ' + words[idx] in bigram_e else 0
        return lang_score(words[idx], vi_score, en_score)


def text_to_phoneme(sentence):
    result = ''
    words = ViTokenizer.tokenize(sentence.strip()).lower().replace('_', ' ').split(' ')

    if len(words) == 1 or (len(words) == 2 and words[-1] in [',', '.', '!', '?']): #fix bug padding
        words += ['.','.','.']

    for idx, word in enumerate(words):
        if word in [',', '.', '!', '?']:
            result += word + ' '
        elif word in en_dict and word not in vi_dict:
            result += en_dict[word] + ' '

        elif word not in en_dict and word not in vi_dict:
            check = True
            for i in word:
                if i in fundamental+'đ':
                    check = False
                    break
            if word in name_dict or check:
                result += '-'.join(g2p(word)).lower() + ' '
            else:
                words_pro = pronounciation(word)
                for w in words_pro:
                    if w in vi_dict:
                        result += word_to_phoneme(w) + ' '
                    elif w == 'đ':
                        result += '-'.join(g2p('d')).lower() + ' '
                    else:
                        result += '-'.join(g2p(w)).lower() + ' '
        elif word in en_dict and word in vi_dict and word in unigram_ambiguous_v:
            result += lang_process(idx, words) + ' '
        else:
            result += word_to_phoneme(word) + ' '
    result = result.replace('\'', '').replace('  ', ' ').strip()
    return result

def metadata_to_phoneme():
    with open('/data/Olli-Speech-1.6/metadata_phoneme_en.csv', 'w') as fw:
        with open('/data/Olli-Speech-1.6/metadata_phoneme_pytorch.txt', 'w') as fw1:
            with open('/data/Olli-Speech-1.6/metadata.csv', 'r') as fr:
                for line in fr.readlines():
                    text = line[:-1].split('|')
                    phoneme = text_to_phoneme(text[2])+'\n'
                    fw.writelines(text[0]+'|'+text[2]+'|'+phoneme)
                    fw1.writelines('/data/Olli-Speech-1.6/wavs/'+text[0]+'.wav|'+phoneme)

#
greetings = [
'high school musical thực chất là một vở nhạc kịch chứ không phải cuộc sống thật của các nhân vật.',
'sau khi trình diễn ca khúc chủ đề five, các cô gái cũng chia sẻ tin vui là bài hát này đã lên vị trí thứ năm trên bảng xếp hạng theo thời gian thực của melon,',
'ngày nay, trên một máy tính có chứa một ổ đĩa cứng có dung lượng năm trăm gigabyte là điều bình thường thì một megabyte chẳng có ý nghĩa gì cả.',
'điều này có lẽ là một fact sự thật, thì đúng hơn là một giả thuyết, cả high school musical và breaking bad đều diễn ra ở albuquerque, new mexico.',
'valentines day còn được gọi là ngày lễ tình yêu hay ngày lễ tình nhân. được đặt tên theo thánh valentine, đây là ngày mà cả thế giới tôn vinh tình yêu đôi lứa.',
'ông già noel mang những đặc điểm kết hợp của vị cha giáng sinh của người anh, thánh sinterklaas của người hà lan và thánh nicholas của người myra ở hy lạp,',
'trang web youtube của google có lượt xem cực khủng. mỗi tháng trang web này đạt tổng số sáu tỷ giờ xem.',
'HLV người Bồ Đào Nha xếp thêm một tiền vệ đánh chặn - Morgan Schneiderlin - bên cạnh Idrissa Gueye nhằm khép chặt trung lộ.',
'blogger là dịch vụ blog miễn phí được cung cấp bởi google. bạn có thể viết blog, tạo website miễn phí trên đây.',
'chiều ngày ba mươi tháng mười hai shane filan thủ lĩnh của ban nhạc đình đám một thời westlife có mặt tại hà nội.',
'mời bạn nghe bài nothing gonna change my love for you do westlife trình bày trên nhạc của tui,',
'ngân hàng a c b mở chi nhánh mới gần f p t quận 2.',
    'mời bạn nghe bài sa mưa dông do hương lan trình bày trên nhạc của tui',
]
# for sent in greetings:
#   print(text_to_phoneme(processSent(sent)))
#metadata_to_phoneme()
print(text_to_phoneme('mời bạn nghe bài the song of the soul được july trình bày trên nhạc của tui'))
#print(word_to_phoneme('-'.join(g2p('vậy')).lower()))

