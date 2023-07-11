import numpy as np
import log
def generate_non_iid_tasks(train_set):


    def group_data_by_class():

        indexs = [[] for _ in range(len(train_set.classes))]  # you can't use `[[]] * len(dataset.classes)`. Although there might be better ways but I don't know
        for idx, (_, class_idx) in enumerate(train_set):
            indexs[class_idx].append(idx)

        assert len(indexs) == 10
        return indexs

    def get_class(row):
          c1, c2 = row[0], row[1]
          return train_set[c1][1],train_set[c2][1]

    num_tasks = 100
    num_classes_per_task = 2
    num_shards = 20
    num_classes = 10

    indexs = group_data_by_class()
    shards = [indexs[i][:num_shards] for i in range(len(indexs))]
    shards = np.array(shards)

    repeated_choices = np.empty((num_classes, num_shards), dtype=int)
    tasks = []

    for i in range(num_shards):
        chosen_elements = np.array([np.random.choice(row, replace=False) for row in shards])
        assert [train_set[j][1] for j in chosen_elements] == list(range(0, 10))
        np.random.shuffle(chosen_elements)
        repeated_choices[:,i] = chosen_elements

    tasks = repeated_choices.T.reshape(num_tasks, num_classes_per_task)

    problem = 0
    for ele in tasks:
      i, j = ele[0], ele[1]
      if train_set[i][1] == train_set[j][1]:
        problem += 1
    assert problem == 0

    classes_per_task = np.apply_along_axis(get_class, axis=1, arr=tasks)
    log.info("Successfully generated {} tasks".format(num_tasks))
    return classes_per_task, indexs