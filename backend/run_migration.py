from migrations.add_pinned_column import upgrade

if __name__ == "__main__":
    try:
        upgrade()
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Error running migration: {str(e)}") 