import torch
from tqdm.auto import tqdm

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    #Set the model to training mode
    model.train()
    #set a variable to track train and test loss
    train_loss, train_acc = 0 , 0

    #training loop
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        #Forward pass
        y_pred = model(X)
        #Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        #optimize
        optimizer.zero_grad()
        #Backpropogation
        loss.backward()
        #perform gradient descent
        optimizer.step()

        #Calculate the train accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        scheduler=None
):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            y_preds = model(X)
            # calculate the loss
            loss = loss_fn(y_preds, y)
            test_loss += loss.item()
            
            y_pred_class = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred_class)
        test_acc = test_acc / len(dataloader)
        test_loss = test_loss / len(dataloader)

        if scheduler is not None:
            scheduler.step(test_loss)
    
    return test_loss, test_acc


def run(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        epochs: int = 5,
        scheduler = None
):
    result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
    
        train_loss, train_acc = train_step(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            
        )

        test_loss, test_acc = test_step(
            model= model,
            dataloader = dataloader,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler
        )

        print(f"Epoch:{epoch}Train Loss:{train_loss:.4f}\tTrain Acc:{train_acc:.4f}\tTest Loss:{test_loss:.4f}\tTest Acc:{test_acc:.4f}")
        result["test_acc"].append(test_acc)
        result["test_loss"].append(test_loss)
        result["train_acc"].append(train_acc)
        result["train_loss"].append(train_loss)
    
    return result
