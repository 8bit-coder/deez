#Resize tensors
model = torch.load("consolidated.00.pth", map_location=torch.device('cpu'))
x = model["tok_embeddings.weight"]
y = model["output.weight"]
row_exclude = 32000
x = x[:row_exclude]
y = y[:row_exclude]
model["tok_embeddings.weight"] = x
model["output.weight"] = y
torch.save(model, "consolidated.01.pth")
#Delete consolidated.00.pth and rename consolidated.01.pth into consolidated.00.pth