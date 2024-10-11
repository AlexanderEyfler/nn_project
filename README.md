# nn_project

First project on working with neural networks, completed during studies at https://github.com/Elbrus-DataScience

  

## Team composition

* Данила Бондаренко https://github.com/dnlbond

* Диана Высоцкая https://github.com/Xurri

* Александр Эйфлер https://github.com/AlexanderEyfler

  

# Installing the environment on a local computer

### 1. Клонировать репозиторий и перейти в директорию проекта

```
git clone https://github.com/AlexanderEyfler/nn_project.git
cd nn_project
```

### 2. Создать окружение из environment.yml

```
conda env create --prefix ./env -f environment.yml
```

Это создаст окружение в директории ./env внутри вашего проекта с указанными зависимостями.

### 3. Активировать окружение

```
conda activate ./env
```

Обратите внимание, что вы указываете путь к окружению, а не его имя.

### 4. Опционально, обновления зависимостей

Если кто-то из участников обновит зависимости в файле environment.yml, то

установить их себе можно с помощью команды:

```
conda env update --prefix ./env -f environment.yml --prune
```

### 5. Общие замечания

* Окружение не коммитится в репозиторий:

Убедитесь, что директория ./env добавлена в .gitignore, чтобы избежать загрузки больших и ненужных файлов в GitHub.

* Участники создают свое локальное окружение:

Каждый участник должен выполнить команду `conda env create --prefix ./env -f environment.yml` в корневой директории проекта, чтобы создать свое собственное окружение.

* Активирование окружения:

Все участники будут использовать команду `conda activate ./env` для активации окружения.

* Относительные пути:

Использование относительных путей при активации окружения (`conda activate ./env`) гарантирует, что команды будут работать одинаково на Windows, macOS и Linux.

* Проблемы с правами доступа:

На некоторых системах могут возникнуть проблемы с правами доступа при создании окружения в определенных директориях. Убедитесь, что у вас есть необходимые права.