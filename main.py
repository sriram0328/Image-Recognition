import flet as ft

def main(page: ft.Page):
    page.title = "Image Recognition"
    page.add(ft.Text("Hello, Flet!"))

ft.app(target=main, view=ft.WEB_BROWSER)  # This ensures it runs as a web app
