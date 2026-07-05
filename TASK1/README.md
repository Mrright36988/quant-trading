# TASK1 量化交易作业

本仓库包含 TASK1 作业相关文件：

- `fetch_tushare_stock.py`：Tushare 数据获取与收盘价绘图脚本。
- `000776_SZ_daily.csv`：广发证券（000776.SZ）过去一年交易日数据。
- `000776_SZ_close.png`：每日收盘价曲线图。

运行脚本时需要自行提供 Tushare token：

```bash
export TUSHARE_TOKEN="your token"
python fetch_tushare_stock.py --ts-code 000776.SZ --name 广发证券
```

说明：当前 token 未开通 Tushare Pro `daily` 接口权限时，脚本会降级使用 Tushare legacy `get_k_data` 接口获取日线数据。
