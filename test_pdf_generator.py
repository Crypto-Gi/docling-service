#!/usr/bin/env python3
"""
Generate a complex test PDF for docling-service testing.
Includes: tables, images, headings, lists, and various formatting.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image as PILImage
import random

def create_sample_image(width=200, height=150, color='blue'):
    """Create a simple colored rectangle image."""
    img = PILImage.new('RGB', (width, height), color=color)
    # Add some pattern
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for i in range(0, width, 20):
        draw.line([(i, 0), (i, height)], fill='white', width=2)
    for i in range(0, height, 20):
        draw.line([(0, i), (width, i)], fill='white', width=2)
    
    # Save to BytesIO
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def generate_complex_pdf(filename='test_document.pdf'):
    """Generate a complex PDF with various elements."""
    
    # Create PDF
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
    )
    
    # Title
    elements.append(Paragraph("Docling Service Test Document", title_style))
    elements.append(Spacer(1, 12))
    
    # Introduction
    intro_text = """
    This is a comprehensive test document designed to evaluate the Docling service's 
    ability to convert complex PDFs to Markdown. It includes various elements such as 
    tables, images, headings, lists, and formatted text.
    """
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Section 1: Financial Data Table
    elements.append(Paragraph("1. Financial Summary Table", heading_style))
    
    financial_data = [
        ['Month', 'Revenue', 'Expenses', 'Profit', 'Growth %'],
        ['January', '$45,230', '$32,100', '$13,130', '12.5%'],
        ['February', '$52,890', '$35,200', '$17,690', '16.8%'],
        ['March', '$61,450', '$38,900', '$22,550', '16.3%'],
        ['April', '$58,320', '$36,500', '$21,820', '-5.1%'],
        ['May', '$67,890', '$41,200', '$26,690', '16.4%'],
        ['June', '$72,100', '$43,800', '$28,300', '6.2%'],
        ['Total', '$357,880', '$227,700', '$130,180', 'Avg: 10.5%'],
    ]
    
    financial_table = Table(financial_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    financial_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8F8F5')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(financial_table)
    elements.append(Spacer(1, 20))
    
    # Section 2: Product Inventory
    elements.append(Paragraph("2. Product Inventory Status", heading_style))
    
    inventory_data = [
        ['Product ID', 'Name', 'Category', 'Stock', 'Unit Price', 'Total Value'],
        ['PRD-001', 'Laptop Pro 15"', 'Electronics', '45', '$1,299.00', '$58,455.00'],
        ['PRD-002', 'Wireless Mouse', 'Accessories', '230', '$29.99', '$6,897.70'],
        ['PRD-003', 'USB-C Hub', 'Accessories', '120', '$49.99', '$5,998.80'],
        ['PRD-004', 'Monitor 27"', 'Electronics', '67', '$399.00', '$26,733.00'],
        ['PRD-005', 'Keyboard Mech', 'Accessories', '89', '$129.99', '$11,569.11'],
    ]
    
    inventory_table = Table(inventory_data, colWidths=[1*inch, 1.3*inch, 1.1*inch, 0.8*inch, 1*inch, 1.2*inch])
    inventory_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(inventory_table)
    elements.append(Spacer(1, 20))
    
    # Section 3: Image
    elements.append(Paragraph("3. Sample Chart/Diagram", heading_style))
    
    # Create and add sample image
    img_buffer = create_sample_image(300, 200, 'lightblue')
    img = Image(img_buffer, width=3*inch, height=2*inch)
    elements.append(img)
    elements.append(Spacer(1, 20))
    
    # Section 4: Bullet Points
    elements.append(Paragraph("4. Key Features & Benefits", heading_style))
    
    features = [
        "• <b>High Performance:</b> Process documents 3x faster with GPU acceleration",
        "• <b>Multi-Format Support:</b> PDF, Word, Excel, PowerPoint conversions",
        "• <b>Cloud Integration:</b> Automatic upload to Cloudflare R2 storage",
        "• <b>AI-Ready Output:</b> Clean Markdown optimized for LLM processing",
        "• <b>REST API:</b> Easy integration with existing workflows",
        "• <b>Real-time Progress:</b> Track conversion status with polling endpoints",
    ]
    for feature in features:
        elements.append(Paragraph(feature, styles['Normal']))
        elements.append(Spacer(1, 6))
    elements.append(Spacer(1, 14))
    
    # Page Break
    elements.append(PageBreak())
    
    # Section 5: Complex Nested Table
    elements.append(Paragraph("5. Employee Performance Matrix", heading_style))
    
    performance_data = [
        ['Employee', 'Q1 Score', 'Q2 Score', 'Q3 Score', 'Q4 Score', 'Average', 'Rating'],
        ['Alice Johnson', '92', '95', '88', '91', '91.5', 'Excellent'],
        ['Bob Smith', '78', '82', '85', '87', '83.0', 'Good'],
        ['Carol White', '95', '97', '96', '98', '96.5', 'Outstanding'],
        ['David Brown', '71', '75', '73', '76', '73.8', 'Satisfactory'],
        ['Eve Davis', '88', '90', '92', '89', '89.8', 'Very Good'],
    ]
    
    perf_table = Table(performance_data, colWidths=[1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.1*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        # Highlight high performers
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#D5F4E6')),  # Carol
        ('TEXTCOLOR', (0, 3), (-1, 3), colors.HexColor('#117A65')),
    ]))
    
    elements.append(perf_table)
    elements.append(Spacer(1, 20))
    
    # Section 6: Code Block (as text)
    elements.append(Paragraph("6. Sample API Request", heading_style))
    
    code_text = """
    <font name="Courier" size="9">
    curl -X POST http://localhost:5010/api/convert \\<br/>
    &nbsp;&nbsp;-F "file=@document.pdf"<br/>
    <br/>
    Response:<br/>
    {<br/>
    &nbsp;&nbsp;"task_id": "a1b2c3d4e5f6"<br/>
    }
    </font>
    """
    elements.append(Paragraph(code_text, styles['Code']))
    elements.append(Spacer(1, 20))
    
    # Footer
    footer_text = """
    <i>This document was automatically generated for testing purposes. 
    It contains sample data and should not be used for actual business decisions.</i>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ Generated: {filename}")
    return filename

if __name__ == "__main__":
    generate_complex_pdf()
