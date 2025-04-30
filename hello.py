from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("dataset/testing_data/images/82092117.png")
print(result.document.export_to_markdown())
