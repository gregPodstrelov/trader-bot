# Strategy lab results

Runs: 27   settled windows with book data: 3061   snapshots: 2,430,203   span: 2026-11-11 18:40:10 -> 2026-11-22 11:44:27 UTC
YES base rate: 0.479   median |model - mid|: 2.5c

```

## 0. Data check: is the Kalshi book live?
           snapshots  windows  %snapshots w/ quote change  median spread (c)  median ask size
series                                                                                       
KXBTC15M      493286      612                        57.0                1.0           4781.4
KXDOGE15M     452655      613                        68.5                1.0             74.5
KXETH15M      494074      612                        65.6                1.0            449.7
KXSOL15M      494827      612                        69.0                1.0            244.4
KXXRP15M      495361      612                        67.6                1.0            259.0
(The old cached data changed quotes on ~3% of snapshots. Live data should be far higher.)

## 1. Model calibration
(one snapshot per window per minute)
                 n  predicted  actual
fv                                   
(0.0, 0.05]   4701      0.014   0.028
(0.05, 0.15]  3685      0.099   0.127
(0.15, 0.3]   5346      0.226   0.248
(0.3, 0.45]   6920      0.380   0.412
(0.45, 0.55]  5418      0.498   0.522
(0.55, 0.7]   5552      0.621   0.630
(0.7, 0.85]   4091      0.771   0.810
(0.85, 0.95]  2526      0.901   0.919
(0.95, 1.0]   3398      0.986   0.983

## 2. Kalshi price calibration
(one snapshot per window per minute)
                 n  predicted  actual
mid                                  
(0.0, 0.05]   4122      0.017   0.015
(0.05, 0.15]  3637      0.098   0.114
(0.15, 0.3]   5512      0.226   0.237
(0.3, 0.45]   6268      0.379   0.391
(0.45, 0.55]  5420      0.500   0.503
(0.55, 0.7]   5619      0.622   0.595
(0.7, 0.85]   4374      0.773   0.773
(0.85, 0.95]  2924      0.902   0.913
(0.95, 1.0]   3763      0.983   0.989

## 3. Repricing speed: BTC moves >= 5 bps in 1s
% of the model's fair-value jump that the Kalshi mid has absorbed after N seconds
series          0s     1s     2s     3s     5s    10s    20s     n   edge left at +1s after fee (median, c)
KXBTC15M       44%    91%    91%    91%    90%    88%    94%   769   -0.7
KXETH15M       47%    99%    97%    97%    95%    95%   100%   716   -1.6
KXSOL15M       47%   103%   105%   106%   103%   101%   100%   608   -2.2
KXXRP15M       46%   103%   104%   103%    98%    98%   103%   634   -1.9
KXDOGE15M      40%   106%   110%   106%   104%    98%    94%   414   -3.8

## 4a. Take the gap, any time in the window
edge>=  trades  win%  avg_fill  avg_model  P&L/contract (±1 s.e.)  total $ @10/trade  t-stat
   2c    5538   50%      0.50       0.51    -1.48c ±0.64            -820.45    -2.3
   3c    5290   50%      0.50       0.52    -1.57c ±0.65            -828.17    -2.4
   5c    4588   50%      0.49       0.53    -1.28c ±0.69            -588.76    -1.9
   8c    3336   47%      0.48       0.54    -2.10c ±0.80            -700.48    -2.6
  12c    2055   44%      0.43       0.54    -1.16c ±1.00            -239.34    -1.2

  Breakdown at the strongest threshold (12c):
  by coin:  BTC -2.7c (n=390)  DOGE -1.6c (n=652)  ETH -1.6c (n=355)  SOL +2.7c (n=330)  XRP -1.9c (n=328)
  by day:   4 of 12 days positive; worst day -7.3c/contract, best +3.7c
  11-11:-3c 11-12:-3c 11-13:-2c 11-14:-1c 11-15:+1c 11-16:+4c 11-17:+1c 11-18:-4c 11-19:-2c 11-20:+2c 11-21:-3c 11-22:-7c

## 4b. Last 3 minutes only
edge>=  trades  win%  avg_fill  avg_model  P&L/contract (±1 s.e.)  total $ @10/trade  t-stat
   2c    2748   41%      0.41       0.45    -1.31c ±0.72            -361.01    -1.8
   3c    2484   39%      0.39       0.44    -1.44c ±0.77            -356.51    -1.9
   5c    2090   37%      0.38       0.44    -1.71c ±0.87            -356.77    -2.0
   8c    1625   36%      0.36       0.45    -1.16c ±1.02            -187.97    -1.1

  Breakdown at the strongest threshold (8c):
  by coin:  BTC +0.6c (n=301)  DOGE -2.5c (n=374)  ETH -1.3c (n=344)  SOL +0.3c (n=289)  XRP -2.3c (n=317)
  by day:   4 of 12 days positive; worst day -8.0c/contract, best +2.6c
  11-11:-0c 11-12:-4c 11-13:-1c 11-14:-1c 11-15:-2c 11-16:+2c 11-17:+3c 11-18:-8c 11-19:+1c 11-20:+2c 11-21:-4c 11-22:-0c

## 4c. Mid window only (3-12 min left)
edge>=  trades  win%  avg_fill  avg_model  P&L/contract (±1 s.e.)  total $ @10/trade  t-stat
   3c    4731   50%      0.50       0.53    -2.00c ±0.66            -945.13    -3.0
   5c    3856   50%      0.50       0.54    -1.56c ±0.74            -602.13    -2.1
   8c    2553   47%      0.48       0.55    -2.63c ±0.91            -672.23    -2.9

## 4d. Long shots only (fill price <= 20c)
edge>=  trades  win%  avg_fill  avg_model  P&L/contract (±1 s.e.)  total $ @10/trade  t-stat
   3c    4060   46%      0.46       0.51    -1.59c ±0.59            -645.85    -2.7
   5c    3265   43%      0.43       0.49    -1.26c ±0.65            -410.15    -1.9
   8c    2107   36%      0.36       0.45    -0.69c ±0.82            -146.11    -0.8
```

How to read: fills use the book ~1s after the signal, with the taker fee. An edge is believable only with t-stat > 2 AND most days positive.
