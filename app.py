# главная страница веб приложения
import streamlit as st

st.title("Многостраничное Приложение Streamlit")

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к странице:", ["Кофейные зёрна", "Сельскохозяйственные культуры", "Случайные изображения"])


# cюда не смотрите, пока не работает и может не будет
if selection == "Кофейные зёрна":
    pass
    import pages.coffee_beans
    pages.coffee_beans.main()
elif selection == "Сельскохозяйственные культуры":
    pass
    import pages.agricultural_crops
    pages.agricultural_crops.main()
elif selection == "Случайные изображения":
    import pages.random_images
    pages.random_images.main()