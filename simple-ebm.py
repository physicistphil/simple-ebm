import torch
import numpy as np
import matplotlib.pyplot as plt
from data import sample_2d_data


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        layer1 = torch.nn.Linear(2, 128)
        act1 = torch.nn.SiLU()
        layer2 = torch.nn.Linear(128, 128)
        act2 = torch.nn.SiLU()
        layer3 = torch.nn.Linear(128, 128)
        act3 = torch.nn.SiLU()
        layer4 = torch.nn.Linear(128, 128)
        act4 = torch.nn.SiLU()
        layer5 = torch.nn.Linear(128, 128)
        act5 = torch.nn.SiLU()
        layer6 = torch.nn.Linear(128, 128)
        act6 = torch.nn.SiLU()
        layer7 = torch.nn.Linear(128, 1)
        self.layers = torch.nn.ModuleList([layer1,
                                           act1,
                                           layer2,
                                           act2,
                                           layer3,
                                           act3,
                                           layer4,
                                           act4,
                                           # layer5,
                                           # act5,
                                           # layer6,
                                           # act6,
                                           layer7
                                           ])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
        return x


class ReplayBuffer():
    def __init__(self, max_size, init_data):
        self.sample_list = init_data
        self.max_size = max_size

    # This starts populated with noise and then is eventually replaced with generated samples
    def add(self, samples):
        self.sample_list = np.concatenate([self.sample_list, samples.numpy()], axis=0)
        buffer_len = self.sample_list.shape[0]
        if buffer_len > self.max_size:
            self.sample_list = np.delete(self.sample_list, np.s_[0:buffer_len - self.max_size], 0)

    def sample(self, num_samples):
        buffer_len = self.sample_list.shape[0]
        indicies = np.random.randint(0, buffer_len,
                                     num_samples if buffer_len > num_samples else buffer_len)
        return self.sample_list[indicies]


model = NeuralNet().cuda()
# data = np.random.multivariate_normal([0, 0], [[1, 5], [5, 10]], size=1000)
data = sample_2d_data('rings', 4096)
dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                         batch_size=128, shuffle=True, num_workers=12)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)

replay_buffer = ReplayBuffer(8192, np.random.randn(*data.shape))

num_epochs = 2000
reg_amount = 1
replay_frac = 0.95


def sample_langevin(x):
    noise_scale = 0.005
    sample_steps = 10
    step_size = 10

    sample_list = []
    sample_list.append(x.detach())
    for _ in range(sample_steps):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        # I'm a little confused on the mechanics of this step.
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size + noise
        sample_list.append(x.detach().cpu())
    return sample_list[-1]


for epoch in range(num_epochs):
    total_loss = []
    # total_poss_energy = []
    for pos_x, in dataloader:
        pos_x = torch.Tensor(pos_x).cuda()
        # pos_x.requires_grad = True
        batch_size = pos_x.shape[0]

        neg_x = replay_buffer.sample(int(batch_size * replay_frac))
        neg_x_rand = np.random.randn(batch_size - neg_x.shape[0], *list(pos_x.shape[1:]))
        neg_x = np.concatenate([neg_x, neg_x_rand], axis=0)
        neg_x = torch.Tensor(neg_x).cuda()
        neg_x.requires_grad = True

        neg_x = sample_langevin(neg_x)
        replay_buffer.add(neg_x)

        optimizer.zero_grad()

        pos_energy = model(pos_x)
        neg_energy = model(neg_x.cuda())
        energy_regularization = reg_amount * (pos_energy.square() +
                                              neg_energy.square())
        loss = (pos_energy - neg_energy + energy_regularization).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm(loss, 0.01)
        total_loss.append(loss.item())
        # print("Batch loss: {}".format(loss))

        optimizer.step()

    print("Epoch: {}\t Loss: {}".format(epoch, np.mean(total_loss)))

rand_num = torch.randn((1000, 2), requires_grad=True)
samples = sample_langevin(rand_num)

plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], marker='+', s=2)
plt.scatter(samples[:, 0], samples[:, 1], marker='x', s=2)
plt.savefig("img/data+samples.png")

plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], marker='+', s=2)
plt.savefig("img/data.png")

plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], marker='x', s=2)
plt.savefig("img/samples.png")
