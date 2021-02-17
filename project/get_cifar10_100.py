import data.cifar_edited as ce

if __name__ == "__main__":
    dataset100 = ce.Cifar100(128, 2)
    dataset10 = ce.Cifar10(128, 2)
    count10 = 0
    count100 = 0
    cisla = []
    for prvok10, prvok100 in zip(dataset10.test.dataset.targets, dataset100.test.dataset.targets):
        if prvok10 == 0: count10 += 1
        if prvok100 == 0: count100 += 1

    print(count10)
    print(count100)

