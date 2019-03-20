from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE(nn.Module):
    def __init__(self, ngf, ndf, nc, nz, have_cuda):
        super(VAE, self).__init__()

        self.have_cuda = have_cuda

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Flatten()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, ngf * 8, 8, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 3, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Change input size in fc1 to your flattened output
        self.fc1 = nn.Linear(82944, 512)
        
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv)
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=3):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    the_loss = BCE + beta*KLD
    return the_loss

def validation(model, testloader):
    test_loss = 0
    for data in testloader:
        images, _ = data
        images = images.cuda()
        decoded, mu, logvar = model.forward(images)
        test_loss += loss_function(decoded, images, mu, logvar).item()
    return test_loss

def save_im(tensor, title, directory):
    image = tensor.cpu().clone()
    x = image.clamp(0, 255)
    x = x.view(x.size(0), 3, 96, 96)
    img_name = "image_{}.png".format(title)
    save_image(x, directory + img_name)
    

def train_vae(model, optim, print_every, num_epochs, trainloader, testloader, image_save_directory = "VAE_IMAGES/"):
    running_loss = 0
    steps = 0
    decoded = 0
    for epoch in range(num_epochs):
        model.train()
        for data in trainloader:
            steps += 1
            img, _ = data
            img = img.cuda()
        
            decoded, mu, logvar = model(img)
            loss = loss_function(decoded, img, mu, logvar)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss = validation(model, testloader)
            
                model.train()
            
                print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
                      "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                      "valid Loss: {:.4f}.. ".format(valid_loss/len(testloader)))
                running_loss = 0
        if epoch % 10 == 0:
            save_im(decoded, 'epoch '+str(epoch), image_save_directory)