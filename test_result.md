## Docling Service Test Document

This is a comprehensive test document designed to evaluate the Docling service's ability to convert complex PDFs to Markdown. It includes various elements such as tables, images, headings, lists, and formatted text.

## 1. Financial Summary Table

![Image](https://pdf2md.mynetwork.ing/images/bc50a7660e0040c395106bd11821106b/picture-1.png)

| Month    | Revenue   | Expenses   | Profit   | Growth %   |
|----------|-----------|------------|----------|------------|
| January  | $45,230   | $32,100    | $13,130  | 12.5%      |
| February | $52,890   | $35,200    | $17,690  | 16.8%      |
| March    | $61,450   | $38,900    | $22,550  | 16.3%      |
| April    | $58,320   | $36,500    | $21,820  | -5.1%      |
| May      | $67,890   | $41,200    | $26,690  | 16.4%      |
| June     | $72,100   | $43,800    | $28,300  | 6.2%       |
| Total    | $357,880  | $227,700   | $130,180 | Avg: 10.5% |

## 2. Product Inventory Status

| Product ID   | Name           | Category    |   Stock | Unit Price   | Total Value   |
|--------------|----------------|-------------|---------|--------------|---------------|
| PRD-001      | Laptop Pro 15" | Electronics |      45 | $1,299.00    | $58,455.00    |
| PRD-002      | Wireless Mouse | Accessories |     230 | $29.99       | $6,897.70     |
| PRD-003      | USB-C Hub      | Accessories |     120 | $49.99       | $5,998.80     |
| PRD-004      | Monitor 27"    | Electronics |      67 | $399.00      | $26,733.00    |
| PRD-005      | Keyboard Mech  | Accessories |      89 | $129.99      | $11,569.11    |

## 3. Sample Chart/Diagram

![Image](https://pdf2md.mynetwork.ing/images/bc50a7660e0040c395106bd11821106b/picture-2.png)

## 4. Key Features &amp; Benefits

- High Performance:

Process documents 3x faster with GPU acceleration

- Multi-Format Support:

PDF, Word, Excel, PowerPoint conversions

- Cloud Integration:

Automatic upload to Cloudflare R2 storage

- AI-Ready Output:

Clean Markdown optimized for LLM processing

•

REST API:

Easy integration with existing workflows

• Real-time Progress:

Track conversion status with polling endpoints

## 5. Employee Performance Matrix

| Employee      |   Q1 Score |   Q2 Score |   Q3 Score |   Q4 Score |   Average | Rating       |
|---------------|------------|------------|------------|------------|-----------|--------------|
| Alice Johnson |         92 |         95 |         88 |         91 |      91.5 | Excellent    |
| Bob Smith     |         78 |         82 |         85 |         87 |      83   | Good         |
| Carol White   |         95 |         97 |         96 |         98 |      96.5 | Outstanding  |
| David Brown   |         71 |         75 |         73 |         76 |      73.8 | Satisfactory |
| Eve Davis     |         88 |         90 |         92 |         89 |      89.8 | Very Good    |

## 6. Sample API Request

```
curl -X POST http://localhost:5010/api/convert \ -F "file=@document.pdf" Response: { "task_id": "a1b2c3d4e5f6" }
```

This document was automatically generated for testing purposes. It contains sample data and should not be used for actual business decisions.