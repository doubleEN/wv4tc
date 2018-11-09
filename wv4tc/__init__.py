#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/9
# desc: CLI


import sys
from wv4tc.model_eval import seg_predict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("参数非法：arg[1]=model_file arg[2]=unk_doc")
        sys.exit(1)
    label = seg_predict(sys.argv[2], sys.argv[1])
    print(label)
