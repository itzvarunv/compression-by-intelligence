| text | chars | tokens | raw | gzip | xz | fixed | gamma | arith | ideal | w | lossless |
|---|---|---|---|---|---|---|---|---|---|---|---|
| code.txt | 1193 | 556 | 8.00 | 4.20 | 4.80 | 7.05 | 2.10 | 2.05 | 2.01 | 15 | yes |
| gibberish.txt | 1500 | 854 | 8.00 | 4.98 | 5.53 | 8.59 | 6.66 | 5.02 | 4.98 | 15 | yes |
| lecture.txt | 1371 | 327 | 8.00 | 4.18 | 4.85 | 3.63 | 1.36 | 1.17 | 1.13 | 15 | yes |
| wiki.txt | 1305 | 274 | 8.00 | 4.57 | 5.20 | 2.78 | 1.09 | 0.94 | 0.89 | 13 | yes |

Values are bits per character (raw ASCII = 8). 'ideal' = sum of -log2 p / chars. 'w' = fixed-width bits per token, set by the largest r.

| text | enc rank ms/token | enc arith ms/token | dec rank ms/token | dec arith ms/token |
|---|---|---|---|---|
| code.txt | 25.0 | 22.1 | 25.2 | 22.1 |
| gibberish.txt | 26.0 | 25.0 | 39.1 | 30.6 |
| lecture.txt | 24.9 | 21.6 | 24.8 | 21.6 |
| wiki.txt | 27.2 | 21.3 | 24.5 | 21.8 |
