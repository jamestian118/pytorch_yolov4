import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FaceDataset
from model import YOLOv4
from utils import plot_losses
from tqdm import tqdm

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = FaceDataset(args.data_path, img_size=args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize the model
    model = YOLOv4(num_classes=len(train_dataset.classes)).to(device)
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))

    # Set optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for imgs, targets in tqdm(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)

            # Forward pass
            loss, _ = model(imgs, targets)
            epoch_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        losses.extend(epoch_losses)
        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {sum(epoch_losses) / len(epoch_losses)}')

    # Save the trained model
    torch.save(model.state_dict(), f'{args.output_path}/model.pth')
    plot_losses(losses, args.output_path)
